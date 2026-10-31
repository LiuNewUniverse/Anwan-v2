#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安晚一体化统一模型 v2。用法、环境与更新日志见 README.md / CHANGELOG.md。"""

import argparse
import difflib
import json
import math
import os
import re
import shutil
import sys
import time
import types
import zlib
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==================== 配置（原 config.py + A10 各端兼容目录系统） ====================

# ==================== 自动目录检测（各端兼容） ====================
def get_base_dir():
    """
    自动检测安晚的数据目录 BASE_DIR（跨平台通用，移植自老版 A10；5 级回退，选择第一个可写目录）：
      第 1 级：脚本所在目录（非打包/非临时环境）—— 单文件便携优先，data/models 与 train_data 跟随单文件
      第 2 级：/storage/emulated/0/Anwan/（Android 0 目录，Termux 最常用）
      第 3 级：Windows -> C:\\Users\\用户名\\Documents\\Anwan
      第 4 级：Linux/Mac -> ~/Documents/Anwan 与 ~/.local/share/Anwan
      第 5 级：回退：当前工作目录
    返回第一个通过可写测试的目录；全部不可写时回退当前工作目录。
    """
    levels = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    is_pyinstaller = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    # 临时目录判定原来用 `"tmp" in script_dir.lower()`，整串子串匹配过宽：
    # 用户目录里只要带 "tmp" 字样（如 ~/tmpfiles、C:\temp\app）就会被误判成
    # 临时解包目录，第 1 级被跳过，数据目录静默漂移到 XDG/文档目录。
    # 收紧为：PyInstaller 的 _MEI 目录、脚本目录本身就是系统临时目录、
    # 或目录名以 tmp 开头（tmpXXXX 这类 mkdtemp 产物）。
    import tempfile as _tf
    is_temp_dir = ("_MEI" in script_dir
                   or os.path.abspath(script_dir) == os.path.abspath(_tf.gettempdir())
                   or os.path.basename(script_dir).startswith("tmp"))

    # 第 1 级：脚本所在目录（单文件便携优先；跳过 PyInstaller 冻结包与临时解包目录）
    if not is_pyinstaller and not is_temp_dir:
        levels.append((1, "脚本所在目录", script_dir))

    # 第 2 级：手机0目录（Termux/Android 最常用）
    sdcard0 = "/storage/emulated/0"
    if os.path.exists(sdcard0) and os.access(sdcard0, os.W_OK):
        levels.append((2, "Android 0 目录", os.path.join(sdcard0, "Anwan")))

    # 第 3 级：Windows -> C:\Users\用户名\Documents\Anwan
    if sys.platform == "win32":
        home = os.path.expanduser("~")
        docs = os.path.join(home, "Documents")
        if os.path.exists(docs):
            levels.append((3, "Windows 文档目录", os.path.join(docs, "Anwan")))

    # 第 4 级：Linux/Mac -> ~/Documents/Anwan 与 ~/.local/share/Anwan
    else:
        home = os.path.expanduser("~")
        docs = os.path.join(home, "Documents")
        if os.path.exists(docs):
            levels.append((4, "用户文档目录", os.path.join(docs, "Anwan")))
        local_share = os.path.join(home, ".local", "share", "Anwan")
        levels.append((4, "XDG 数据目录", local_share))

    # 第 5 级：当前工作目录（兜底）
    levels.append((5, "当前工作目录", os.getcwd()))

    # 依次尝试，选择第一个可写的目录（跳过重复候选）
    seen = set()
    for level, label, path in levels:
        if path in seen:
            continue
        seen.add(path)
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return path
        except (OSError, PermissionError):
            print(f"  [!] 第 {level} 级目录不可写，自动回退下一级：{label} -> {path}")
            continue

    return os.getcwd()



"""
安晚一体化模型 v2 —— 全局配置
所有可调超参集中于此：词表 32000、情感 512 维、模型尺寸、分段训练、router 自动分区、推理参数全部配置化。
"""


# ==================== 路径 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = get_base_dir()                                 # 各端兼容目录系统（A10，5 级回退）
DATA_ROOT = os.path.join(BASE_DIR, "data")                # 数据落点（词表/checkpoint）
MODEL_DIR = os.path.join(DATA_ROOT, "models")             # checkpoint 目录
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")      # 词表持久化文件
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "anwan_v2.pt")        # 模型检查点
TRAIN_FOLDER = os.path.join(BASE_DIR, "train_data")       # 外部语料文件夹（TRAIN_FOLDER，DataPipeline 读）
LEARN_FOLDER_NAME = "训练数据"                             # learn 渐进式阅读的文件夹名
LEARN_FOLDER = os.path.join(BASE_DIR, LEARN_FOLDER_NAME)  # 学习文件夹（TrainingDataFolder 读，learn 用）
PROGRESS_FILE_NAME = "anwan_train_progress.json"          # 学习进度文件名
PROGRESS_FILE = os.path.join(BASE_DIR, PROGRESS_FILE_NAME)  # 学习进度（散在 BASE_DIR 根目录）
LEARNED_PAIRS_FILE = os.path.join(TRAIN_FOLDER, "learned_pairs.json")  # learn 学习成果落盘（auto 闭环：DataPipeline 统一取数）
LEARNED_PAIRS_NAME = os.path.basename(LEARNED_PAIRS_FILE)              # 落盘文件名（TrainDataFolder 扫描时排除自身）

# ==================== 分词器 ====================
VOCAB_SIZE = 32000          # 词表大小（方案A(语种子词表)+方案B(全局扩表)融合基线：
MAX_LEN = 80               # 输入最长 token 数（Orin 8G + AMP 后可安全撑到 80，有余量再往上加）
SPECIAL_TOKENS = {
    "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
    "<SEP>": 4, "<MASK>": 5, "<USER>": 6, "<BOT>": 7,
}
BPE_MERGE_CAP = 30000       # BPE 合并轮数上限（32K 词表需要更多合并轮次；语料不足时以实际为准）

# ==================== 模型 (800M 参数规格) ====================
EMBED_DIM = 1536           # 共享 embedding 宽度
LSTM_HIDDEN = 8192         # LSTM-A（内容/无状态）宽度
LSTM_LAYERS = 1
LSTM_B_HIDDEN = 3600      # LSTM-B（情感通道）中间宽度（精确调参使总参数≈800M）
EMOTION_DIM = 512          # 情感维度 (512-d → LSTM 2048 → 512 回传)
NUM_HEADS = 12             # TF-1 检索 / TF-2 生成 注意力头数 (1536÷12=128，整除)
TRANS_LAYERS = 5           # TF-1 编码器层数
DECODER_LAYERS = 4         # TF-2 解码器层数
FF_DIM = 6144              # FFN 中间宽度 (4×1536)
DROPOUT = 0.1
TIE_WEIGHTS = True         # 输出投影与 shared_embedding 共享权重

# ==================== 意图头（一体化内嵌，多任务监督） ====================
INTENTS = [
    "greeting", "farewell", "identity", "state_query",
    "emotion_share", "affection", "teasing", "question",
    "admin_cmd", "egg_trigger", "action_request", "chat",
    # 语言学习 / 生僻词学习（14 意图）。顺序必须与 UnifiedIntentLabeler.INTENTS
    # 完全一致 —— intent_head 的输出维度 = len(INTENTS)，标签索引由
    # INTENT2IDX 按名映射，两边错位会导致训练标签指向错误的 logit。
    "lang_learning", "rare_word",
]
INTENT_LOSS_WEIGHT = 8.0   # 意图 loss 加权 8:1（与旧版语义一致，强化入口语义）
# 情感状态安全阈值：跨 batch 串联后 LSTM 状态可能发散，越界就重置而非毒化整个 epoch
# LSTM 标定，换 SSM 后需用 calibrate-emotion 重标
# 从环境变量读取，缺省值沿用当前 LSTM 标定值
_EMOTION_STATE_MAX_NORM_DEFAULT = 50.0
try:
    EMOTION_STATE_MAX_NORM = float(os.environ.get("ANWAN_EMOTION_STATE_MAX_NORM", str(_EMOTION_STATE_MAX_NORM_DEFAULT)))
except ValueError:
    EMOTION_STATE_MAX_NORM = _EMOTION_STATE_MAX_NORM_DEFAULT

# ---- 情绪方向 loss（塑形项，让情感状态"该动的时候动"）----
# 问题：光把状态串起来，模型只会学出"一直保持平稳"——因为大部分 SFT 语料
# 语气都差不多，情感通道的最优解就是不动。
# 做法：按意图给"唤醒度"打分，要求状态变化幅度与之匹配：
#   高唤醒（teasing/emotion_share/egg_trigger）→ 状态必须明显变化
#   低唤醒（chat/question/state_query）        → 状态必须平稳
# 用相对变化率 ‖Δh‖/‖h_in‖ 而非绝对幅度：状态范数会随对话自然增长，
# 绝对目标会和这个增长打架，hinge 只在越界时惩罚，不干预正常增长。
EMOTION_DIRECTION_WEIGHT = 0.3   # 塑形项权重；相对分离度是 O(1) 量，0.3 已能起作用
# 幅度约束用**相对 batch 均值**的边距，不用绝对阈值（绝对阈值对不上模型
# 天然的变化率尺度，实测会让 loss 恒等于 0，塑形完全失效）。
_EMOTION_AROUSAL_MARGIN_DEFAULT = 0.15
try:
    EMOTION_AROUSAL_MARGIN = float(os.environ.get("ANWAN_EMOTION_AROUSAL_MARGIN", str(_EMOTION_AROUSAL_MARGIN_DEFAULT)))
except ValueError:
    EMOTION_AROUSAL_MARGIN = _EMOTION_AROUSAL_MARGIN_DEFAULT
EMOTION_AROUSAL = {
    "teasing": 1.0, "emotion_share": 1.0, "egg_trigger": 0.8,
    "affection": 0.6, "action_request": 0.5, "farewell": 0.5,
    "greeting": 0.4, "state_query": 0.3, "admin_cmd": 0.3,
    "identity": 0.2, "question": 0.2, "rare_word": 0.2,
    "lang_learning": 0.2, "chat": 0.15,
}

# ---- 显式情感向量（P4：让感情真正有向量）----
# 现状：情感通道是 LSTM-B 的 (h,c) 隐状态，只能靠范数变化率"间接塑形"，
# 没有可解释的语义方向，也拿不出"此刻愉悦度多少"这种可读输出。
# 升级：在 LSTM-B 池化输出上接 PAD 三维语义头（愉悦/唤醒/支配，tanh∈[-1,1]），
# 受意图监督回归、跨轮连续性约束、随状态持久化，并反哺下一轮生成。
EMOTION_VEC_DIM = 3

# ---- MoE 轻量混合专家（decoder 层）----
# 仅在 decoder 启用 MoE：4 个小型专家 + Top-2 路由，每 token 激活 2 个。
# 专家 FFN = 原 FFN 的一半，总参数量可控，Orin 8G 可承受。
NUM_EXPERTS = 4              # 专家数量（Orin 8G 安全上限）
TOP_K_EXPERTS = 2            # Top-2 路由（每 token 只激活 2 个专家）
EXPERT_FF_DIM = 3072         # 单个专家的 FFN 中间宽度（原 FF_DIM 6144 的一半）
MOE_LOAD_BALANCE_WEIGHT = 0.01  # 负载均衡 loss 权重（辅助项，小权重防专家坍缩）
# ==================== v3 半专家化（全域专家 + 情感半专家）====================
# 升级口径：不是再加一个新模块，而是把已有 4 条通道统一换成同一套专家化骨架。
#   · 专家模型（纯 MoE）：内容双向 Transformer / 检索 TF-1 / 生成 TF-2 —— 参数可扩、激活不扩。
#   · 半专家 + 完整模型：仅情感 LSTM-B —— 完整主干永远通电并独占跨轮 (h,c)，
#     半专家只做单轮小幅修正，门控带上限，反不了客。
MOE_SCOPE = os.environ.get("ANWAN_MOE_SCOPE", "all")  # "decoder"=旧行为（仅 TF-2 MoE，稠密）| "all"=全域专家化
# ---- 内容通道：双向专家 Transformer（默认）/ LSTM-A 专家池（回退）----
CONTENT_TF_LAYERS = 2          # 内容编码器层数（双向自注意力，每 token 看见整段上下文）
CONTENT_TF_EXPERTS = 4         # 每层 FFN 专家数
CONTENT_TF_TOP_K = 2           # 每 token 激活专家数
CONTENT_TF_FF_DIM = 1536       # 单专家 FFN 宽度（与 TF-1 检索同级，手机端标准算子）
CONTENT_PACK_MAX_SEGS = 4      # 一次前向最多打包的文本段数
CONTENT_PACK_PROB = 0.4        # LM/SFT 把多段文本打进同一 src 的概率
LSTM_A_EXPERTS = 4             # LSTM 回退：专家个数
LSTM_A_TOP_K = 1               # LSTM 回退：每条序列只激活 1 个专家
LSTM_A_EXPERT_HIDDEN = 4096    # LSTM 回退：单专家隐层
LSTM_A_LOAD_BALANCE_WEIGHT = 0.01
LSTM_A_MIN_LOAD = 0.10            # 单个内容专家的最低平均负载（饥饿下界）
# ---- TF-1 检索编码器：专家化 FFN ----
ENC_EXPERTS = 4
ENC_TOP_K = 2
ENC_EXPERT_FF_DIM = 1536
ENC_LOAD_BALANCE_WEIGHT = 0.01
# ---- 情感通道：半专家 + 完整模型 ----
EMO_FULL_MODEL = True          # 完整模型底座开关（False=退化为纯专家，不建议）
EMO_SEMI_EXPERTS = 2           # 半专家个数
EMO_SEMI_TOP_K = 1             # 每轮每序列激活的半专家数
EMO_SEMI_EXPERT_HIDDEN = 1200  # 半专家隐层（完整 3600 的 1/3）
EMO_SEMI_GATE_INIT = 0.15      # 半专家支路门控初值（先让完整模型学会说话）
EMO_SEMI_GATE_MAX = 0.5        # 门控上限：半专家贡献永远不能超过完整模型
EMO_LOAD_BALANCE_WEIGHT = 0.01
EMO_SEMI_MIN_LOAD = 0.15        # 单个半专家的最低平均负载：防止某个半专家整程被饿死
MIN_LOAD_WEIGHT = 0.5           # 饥饿 hinge 项权重
EMOTION_PAD_LABELS = ("愉悦", "唤醒", "支配")
# 意图 → PAD 目标向量（值域 [-1,1]）。与 EMOTION_AROUSAL 语义一致，但给出具体
# 方向：低唤醒意图目标贴近 0，高唤醒意图目标活跃（兴奋/共情/调侃各有各的位形）。
EMOTION_PAD_TARGETS = {
    "greeting":       (0.40, 0.35, 0.30),
    "farewell":       (0.30, 0.25, 0.10),
    "identity":       (0.20, 0.25, 0.10),
    "state_query":    (0.10, 0.20, 0.05),
    "emotion_share":  (0.00, 0.85, -0.20),   # 共情：唤醒高、愉悦低（承接负面情绪）
    "affection":      (0.85, 0.45, 0.35),    # 亲昵：高愉悦
    "teasing":        (0.35, 0.80, 0.55),    # 调侃：高唤醒 + 支配
    "question":       (0.05, 0.25, 0.00),
    "admin_cmd":      (-0.15, 0.10, 0.45),   # 命令：支配感
    "egg_trigger":    (0.60, 0.90, 0.30),    # 彩蛋：兴奋
    "action_request": (0.45, 0.50, 0.25),
    "chat":           (0.30, 0.20, 0.05),
    "lang_learning":  (0.30, 0.30, 0.00),
    "rare_word":      (0.15, 0.20, -0.10),
}
EMOTION_VEC_WEIGHT = 0.5       # 向量匹配 loss 权重（MSE 量级 ~O(0.1)，0.5 足够显性）
EMOTION_VEC_CONT_WEIGHT = 0.8  # 跨轮连续性 loss 权重（只约束低唤醒意图）
EMOTION_VEC_CONT_AROUSAL_MAX = 0.4  # 唤醒度低于此值的意图才受连续性约束（高唤醒放行跳变）
EMOTION_VEC_INJECT = 1.0       # 上一轮情感向量注入生成的强度系数
# ==================== 内容主干开关 ====================
# tf  = 双向专家 Transformer（默认，手机端标准注意力算子）
# ssm = Mamba2 / 纯 PyTorch S6（需 CUDA 原生核或忍受逐步循环）
# lstm = LSTM-A 专家池 / 稠密 LSTM
ANWAN_BACKBONE = os.environ.get("ANWAN_BACKBONE", "tf")
ANWAN_RETRIEVAL_MODE = os.environ.get("ANWAN_RETRIEVAL_MODE", "full")
ANWAN_EMO_DIR_NORM = os.environ.get("ANWAN_EMO_DIR_NORM", "std")
# 记忆检索：同语种相似度加成（方案A软标记的检索侧落地）
SEARCH_LANG_BONUS = 0.15

# ==================== router 自动分区 ====================
NUM_MODULES = 4            # 内容 Transformer / LSTM-B 情感 / TF-1 检索 / TF-2 生成
ROUTER_ENTROPY_WEIGHT = 0.1  # 熵正则系数（防路由坍缩；>0 鼓励 soft 组合不垄断）
ROUTER_ENTROPY_FLOOR = 0.8   # 熵下限：batch 边际路由熵低于此值才开始惩罚
STAGE = "main"             # "main": 仅对角主通路（内容↔检索、情感↔生成）
                           # "cross": 开放交叉辅通路（内容→生成、情感→检索），弱门控起步
CROSS_GATE_INIT = 0.1      # 交叉辅通路门控初始值（弱门控）
CROSS_GATE_MAX = 0.5       # 交叉辅通路门控上限

# ==================== 训练（分段） ====================
LM_EPOCHS = 30             # 阶段一：LM 预训练（next-token 预测，覆盖全部语料）
SFT_EPOCHS = 30            # 阶段二：监督调优（输入问句 → 生成回复）
BATCH_SIZE = 16
LR = 2e-4
PATIENCE = 15              # 早停耐心
SAVE_EVERY = 25            # checkpoint 周期（轮）
GRAD_CLIP = 1.0
SFT_TRAIN_RATIO = 0.85     # 监督数据的训练/验证切分

# ---- AMP 自动混合精度（Orin Nano 8G 内存优化） ----
AMP_ENABLED = True           # 默认启用 AMP
AMP_DTYPE = "float16"        # float16 | bfloat16（Orin 不支持 bfloat16，保持 float16）

# ---- checkpoint 体积控制（小内存设备必看）----
CKPT_SAVE_SHADOWS = True     # True=连同 EMA/最优快照一起落盘（可续训最优权重）；False=只存主权重

# ---- 概念图谱（跨记忆关联，零依赖）----
GRAPH_TOP_CONCEPTS = 12       # 每条记忆最多保留 N 个概念特征（有区分度者，按权重截断）
GRAPH_SPREAD_DECAY = 0.35     # 一跳激活扩散的衰减系数（0=关闭，0.3~0.5 较稳）
GRAPH_SPREAD_MIN_RATIO = 0.3
GRAPH_CONCEPT_MAX_DF = 0.3  # 概念区分度门槛：特征出现在超过 30% 的记忆里就不算概念

# ---- 检索档位：同一套图谱，按场景切换检索策略 ----
RETRIEVAL_PROFILES = {
    "barrage": {
        "top_k": 1,            # 只注入 1 条，避免回复被无关记忆稀释
        "min_score": 0.32,     # 阈值拉高：宁可不用记忆，也不要用错的
        "spread": False,       # 扩散的价值在"关联深挖"，弹幕没有深挖的时间预算
        "max_tokens": 8,       # 单条注入上限压到 8，腾出上下文给当前弹幕
    },
    "dm": {
        "top_k": 3,
        "min_score": 0.12,
        "spread": True,        # 私聊允许一跳扩散，换取跨记忆关联
        "max_tokens": 15,
    },
}
DEFAULT_PROFILE = "dm"        # 未知场景时的保守档位
BARRAGE_MIN_LEN = 8           # 弹幕模式下短于该长度的消息不入库（绝大多数是噪声）

# ---- AWIL 总线（旁路调试日志）----
AWIL_ENABLED_IN_TRAIN = False

# ==================== 推理 ====================
TEMPERATURE = 0.8
TOP_P = 0.9
GEN_MAX_LEN = 1024         # 生成最长 token 数（动态长度，支持长序列推理）
REPLY_MIN_LEN = 2          # 生成结果低于此长度视为无效，走兜底

# ==================== 多轮历史（P1-1） ====================
HISTORY_TURNS = 4          # respond 保留最近 N 轮显式历史
HISTORY_MAX_TOKENS = 36    # 历史段注入 src 的最大 token 预算（多段打包后给上下文更多位子）
MULTI_TURN_PROB = 0.3      # SFT 样本掺入多轮历史的概率

# ---- P1-2 外部长期记忆（事实/日记类） ----
MEMORY_FILE = os.path.join(MODEL_DIR, "external_memory.json")   # 外部记忆持久化文件
WORDBOOK_FILE = os.path.join(MODEL_DIR, "wordbook.json")         # 生词本（生僻词/外语查询）
MEMORY_TOP_K = 3          # 检索注入条数
MEMORY_MIN_SCORE = 0.10   # 注入的相似度下限（过低不注入，避免噪声）
MEMORY_MAX_TOKENS = 20    # 单条记忆注入 src 的最长 token 数
MEMORY_DECAY = 0.995      # 平滑遗忘：每次触发遗忘时全体活跃记忆权重衰减系数
MEMORY_EXPIRE_WEIGHT = 0.45  # 权重低于此值的低活跃记忆转为"归档"（检索不再返回，可手动恢复）
MEMORY_MAX_ARCHIVE = 200  # 归档上限（超过则真正物理清理最旧归档，防止 json 无限膨胀）
IDF_REBUILD_RATIO = 1.5   # 记忆规模涨到上次重建的 N 倍时重算 IDF 与全部记忆向量

# ---- P1-3 情感状态持久化 ----
EMOTION_STATE_FILE = os.path.join(MODEL_DIR, "emotion_state.pt")  # 跨轮情感 (h,c,emotion_vec) 落盘文件（旧格式兼容加载）
_EMOTION_DECAY_DAYS_DEFAULT = 7.0
try:
    EMOTION_DECAY_DAYS = float(os.environ.get("ANWAN_EMOTION_DECAY_DAYS", str(_EMOTION_DECAY_DAYS_DEFAULT)))
except ValueError:
    EMOTION_DECAY_DAYS = _EMOTION_DECAY_DAYS_DEFAULT

config = types.SimpleNamespace(**{
    k: v for k, v in list(globals().items())
    if isinstance(k, str) and k.isupper() and not k.startswith("_")
})


# ==================== 目录体检与自动归位（放错位置检测） ====================

def _build_layout_rules():
    """按当前模块路径常量构建关键文件落点规则（目录体检用，随常量一并统一变更）。"""
    return [
        {
            "name": "tokenizer.json",
            "memo": "词表",
            "correct": TOKENIZER_FILE,     # BASE_DIR/data/models/tokenizer.json
            "wrong": [
                BASE_DIR,                  # 放到了 BASE_DIR 根目录
                DATA_ROOT,                 # 放到了 data/（应在 data/models/）
                os.path.join(BASE_DIR, "models"),   # 旧版根级 models/
                TRAIN_FOLDER,              # 放到了 train_data/
                LEARN_FOLDER,              # 放到了 训练数据/
            ],
        },
        {
            "name": "anwan_v2.pt",
            "memo": "模型权重",
            "correct": CHECKPOINT_FILE,    # BASE_DIR/data/models/anwan_v2.pt
            "wrong": [
                BASE_DIR,
                DATA_ROOT,
                os.path.join(BASE_DIR, "models"),
                TRAIN_FOLDER,
                LEARN_FOLDER,
            ],
        },
        {
            "name": PROGRESS_FILE_NAME,
            "memo": "学习进度",
            "correct": PROGRESS_FILE,      # BASE_DIR/anwan_train_progress.json（散在根目录）
            "wrong": [
                DATA_ROOT,                 # 放到了 data/
                MODEL_DIR,                 # 放到了 data/models/
                TRAIN_FOLDER,
                LEARN_FOLDER,
            ],
        },
    ]


_LAYOUT_RULES = _build_layout_rules()


def check_and_relocate():
    """
    目录体检：启动时核对关键文件的落点，发现放错位置时提醒用户并自动移动到正确位置。

    检测对象（唯一正确布局）：
      - BASE_DIR/data/models/tokenizer.json（TOKENIZER_FILE）
      - BASE_DIR/data/models/anwan_v2.pt（CHECKPOINT_FILE）
      - BASE_DIR/anwan_train_progress.json（PROGRESS_FILE，散在根目录）

    行为规则：
      - 逐个扫描上述文件的"常见放错位置"（BASE_DIR 根、data/、旧版根级 models/、
        train_data/、训练数据/；当脚本所在目录与 BASE_DIR 不同时也纳入扫描）；
      - 正确位置尚无该文件 → 自动创建目录并移动过去，打印提醒；
      - 正确位置已有同名文件 → 不移动、不覆盖，仅提醒存在多余副本；
      - 扫描范围严格限定在上述有限候选目录内，不影响其它用户文件。
    返回 (moved, warned)。
    """
    script_dir = PROJECT_ROOT
    if script_dir != BASE_DIR:
        for rule in _LAYOUT_RULES:
            if script_dir not in rule["wrong"]:
                rule["wrong"].append(script_dir)

    moved, warned = 0, 0
    for rule in _LAYOUT_RULES:
        fname, memo, correct = rule["name"], rule["memo"], rule["correct"]
        for d in rule["wrong"]:
            p = os.path.join(d, fname)
            if not os.path.isfile(p) or os.path.abspath(p) == os.path.abspath(correct):
                continue
            if os.path.exists(correct):
                print(f"[提醒] 发现多余的「{fname}」副本（{memo}）：{p}")
                print(f"       正确位置 {correct} 已有该文件，未改动该副本；如需清理请手动删除。")
                warned += 1
                continue
            try:
                os.makedirs(os.path.dirname(correct), exist_ok=True)
                shutil.move(p, correct)
                moved += 1
                print(f"[提醒] 检测到文件放错位置：「{fname}」（{memo}）")
                print(f"       原位置: {p}")
                print(f"       已自动移动到正确位置: {correct}")
            except (OSError, PermissionError) as e:
                print(f"[提醒] 「{fname}」（{memo}）自动归位失败：{p} -> {correct}（{e}）")
                print(f"       请手动将该文件移动到: {correct}")
    return moved, warned


# ==================== 分词器（原 anwan/tokenizer.py） ====================

PAD_IDX = config.SPECIAL_TOKENS["<PAD>"]
BOS_IDX = config.SPECIAL_TOKENS["<BOS>"]
EOS_IDX = config.SPECIAL_TOKENS["<EOS>"]
UNK_IDX = config.SPECIAL_TOKENS["<UNK>"]
USER_IDX = config.SPECIAL_TOKENS["<USER>"]   # 多轮历史：用户发言分隔符（P1-1）
BOT_IDX = config.SPECIAL_TOKENS["<BOT>"]     # 多轮历史：安晚回复分隔符（P1-1）
SEP_IDX = config.SPECIAL_TOKENS["<SEP>"]     # 多段文本打包分隔符


class BPEAnwanTokenizer:
    """
    增强型 BPE 分词器
    - 真正的 BPE 子词合并算法
    - 中文词汇优先保留（连续中文片段不拆字）
    - 词表持久化（训练一次，之后启动直接加载）
    - 编码安全回退（越界索引 → <UNK>）
    - 兼容旧版 SimpleTokenizer 接口（train/encode/decode/safe_encode/save_vocab/load_vocab）
    """

    SPECIAL_TOKENS = dict(config.SPECIAL_TOKENS)

    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size or config.VOCAB_SIZE
        self.word2idx = dict(self.SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.next_idx = len(self.SPECIAL_TOKENS)
        # BPE 合并规则：[( (piece_a, piece_b), merged_string ), ...] 按训练顺序
        self.bpe_merges = []
        # 方案A：逐 token 语种归属标签（token 字符串 → 书写系统/语种标签）。
        # 融合方案B(全局32K)后不物理分块，标签只做软分区：语种子预算体检、
        # 记忆检索"同语种优先"、诊断语种饥饿。
        self.token_scripts = {}
        # 词频统计（仅用于诊断/信息展示）
        self.token_freqs = Counter()
        # 预编译正则：多文种连续字符 + 拉丁词 + 数字 + 标点
        # 语言包场景必须覆盖主要书写系统；未列出的文字会落到最后的 \S
        # （逐字符），能工作但词表效率极低，所以这里尽量枚举完整。
        self.pat = re.compile(
            r"[\u4e00-\u9fff\u3400-\u4dbf]+|"          # CJK 表意 + 扩展A
            r"[\u3040-\u30ff\u31f0-\u31ff]+|"          # 日文 假名 + 浊音符
            r"[\uac00-\ud7af]+|"                        # 韩文 音节
            r"[\u0400-\u04ff]+|"                        # 西里尔（俄/乌/保…）
            r"[\u0600-\u06ff]+|"                      # 阿拉伯
            r"[\u0590-\u05ff]+|"                        # 希伯来
            r"[\u0900-\u097f]+|"                        # 天城文（印地）
            r"[\u0e00-\u0e7f]+|"                        # 泰文
            r"[\u0b80-\u0bff]+|"                        # 泰米尔
            r"[\u0c00-\u0c7f]+|"                        # 泰卢固
            r"[\u0370-\u03ff]+|"                        # 希腊
            r"[a-zA-Z]+(?:['’][a-zA-Z]+)*|"          # 拉丁词（保留 don't / it's）
            r"\d+|"
            r"[^\s\w]|"
            r"\S"
        )

    # ---------------- 预分词 ----------------
    def _pre_tokenize(self, text):
        """预分词：把文本切成基础单元（中文整段、英文词、数字、标点）"""
        return self.pat.findall(text)

    # ---------------- 合并索引（encode 加速）----------------
    def _apply_merges(self, word_tokens):
        """对单个词按训练顺序应用 BPE 合并，返回新的 token 列表。
          pair[0] 本身为多字符合并 token 的级联规则，且实测反而更慢，故不采用。）"""
        if len(word_tokens) < 2 or not self.bpe_merges:
            return word_tokens
        present = set(word_tokens)
        for pair, merged in self.bpe_merges:
            if pair[0] not in present:
                continue
            new_tokens = []
            i, n, fired = 0, len(word_tokens), False
            while i < n:
                if i < n - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                    fired = True
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            if fired:
                word_tokens = new_tokens
                present = set(word_tokens)
        return word_tokens

    # ---------------- 语种标签（方案A：软分区）----------------
    def _script_of_token(self, token):
        """单 token 串的语种归属：纯数字/标点 → other；其余按主导书写系统"""
        if not token:
            return "other"
        if token.isascii() and (token.isdigit() or not token.isalpha()):
            return "other"
        return detect_script(token)

    def _tag_token(self, token):
        s = self._script_of_token(token)
        self.token_scripts[token] = s
        return s

    def _ensure_token_scripts(self):
        """旧版词表无标签时补算（加载兼容）"""
        if self.token_scripts:
            return
        for tok in self.word2idx:
            if tok not in self.SPECIAL_TOKENS:
                self.token_scripts[tok] = self._script_of_token(tok)

    def script_of(self, token):
        """查询 token（字符串或 idx）的语种标签"""
        if isinstance(token, int):
            tok = self.idx2word.get(token)
        else:
            tok = token
        if tok is None:
            return "other"
        if tok not in self.token_scripts:
            self.token_scripts[tok] = self._script_of_token(tok)
        return self.token_scripts[tok]

    def vocab_by_script(self, script):
        """某语种下的 token 列表（语种子预算视图）"""
        self._ensure_token_scripts()
        return [t for t, s in self.token_scripts.items() if s == script]

    def script_distribution(self):
        """各语种 token 数分布（体检/预算分配报告用）"""
        self._ensure_token_scripts()
        out = Counter()
        for s in self.token_scripts.values():
            out[s] += 1
        return dict(out)

    # ---------------- 训练 ----------------
    def train(self, texts, num_merges=None):
        """
        真正的 BPE 训练：
        1. 统计所有字符和基础词
        2. 迭代合并最频繁的相邻对
        3. 生成词表和合并规则（满词表自动提前终止）
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.SPECIAL_TOKENS) - 100
        num_merges = min(num_merges, config.BPE_MERGE_CAP)

        # ---- 初始化：每个词拆成字符列表 ----
        vocab = Counter()
        all_chars = set()  # 训练前收集语料全部单字符（合并后单字会从 vocab 消失，兜底必须绕过 vocab）
        for text in texts:
            for word in self._pre_tokenize(text):
                chars = list(word)
                for ch in chars:
                    all_chars.add(ch)
                vocab[tuple(chars)] += 1

        # ---- 增量合并：只重算"受影响的词"，而不是每轮全量重算 pair 频率 ----
        active_vocab = dict(vocab)
        pairs = Counter()
        char_idx = {}          # 字符 -> 含该字符的词集合
        for word_tuple, count in active_vocab.items():
            for j in range(len(word_tuple) - 1):
                pairs[(word_tuple[j], word_tuple[j + 1])] += count
            for ch in set(word_tuple):
                char_idx.setdefault(ch, set()).add(word_tuple)

        progress_every = 2000
        for i in range(num_merges):
            if self.next_idx >= self.vocab_size:
                break
            if not pairs:
                break
            # 确定性排序：频率降序，相同按字典序（与原实现一致）
            best_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
            merged = best_pair[0] + best_pair[1]
            self.bpe_merges.append((best_pair, merged))

            affected = list(char_idx.get(best_pair[0], ()))
            new_vocab = {}
            for word_tuple in affected:
                count = active_vocab.pop(word_tuple)
                for j in range(len(word_tuple) - 1):
                    pairs[(word_tuple[j], word_tuple[j + 1])] -= count
                for ch in set(word_tuple):
                    char_idx.get(ch, set()).discard(word_tuple)
                new_word = []
                j = 0
                while j < len(word_tuple):
                    if (j < len(word_tuple) - 1 and
                            word_tuple[j] == best_pair[0] and
                            word_tuple[j + 1] == best_pair[1]):
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word_tuple[j])
                        j += 1
                nw = tuple(new_word)
                new_vocab[nw] = new_vocab.get(nw, 0) + count
                for j in range(len(nw) - 1):
                    pairs[(nw[j], nw[j + 1])] += count
                for ch in set(nw):
                    char_idx.setdefault(ch, set()).add(nw)
            for word_tuple, count in new_vocab.items():
                if word_tuple in active_vocab:
                    active_vocab[word_tuple] += count
                else:
                    active_vocab[word_tuple] = count

            if merged not in self.word2idx:
                self.word2idx[merged] = self.next_idx
                self.idx2word[self.next_idx] = merged
                self._tag_token(merged)
                self.next_idx += 1
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    合并 {i+1}/{num_merges} | 词表 {self.next_idx} | "
                      f"活跃词 {len(active_vocab)}", flush=True)

        for word_tuple, count in vocab.items():
            for token in word_tuple:
                if token in self.word2idx:
                    self.token_freqs[token] += count

        # 兜底：把语料中出现过的单字符加入词表（合并后低频单字不会丢失）
        if self.next_idx < self.vocab_size:
            for ch in sorted(all_chars):
                if ch in self.word2idx:
                    continue
                if self.next_idx >= self.vocab_size:
                    break
                self.word2idx[ch] = self.next_idx
                self.idx2word[self.next_idx] = ch
                self._tag_token(ch)
                self.next_idx += 1

        print(f"  (｡･ω･｡)ﾉ BPE 合并: {len(self.bpe_merges)} 次 | 词表大小: {self.next_idx} / {self.vocab_size}")

    # ---------------- 编码 ----------------
    def encode(self, text, max_len=None):
        """
        BPE 编码（与训练合并规则一一对应）
        返回固定长度 token 索引列表（不足补 <PAD>，超出截断）
        """
        if max_len is None:
            max_len = config.MAX_LEN
        tokens = []
        for word in self._pre_tokenize(text):
            word_tokens = self._apply_merges(list(word))   # 按训练顺序应用合并规则
            for t in word_tokens:
                tokens.append(self.word2idx.get(t, UNK_IDX))

        if len(tokens) < max_len:
            tokens += [PAD_IDX] * (max_len - len(tokens))
        return tokens[:max_len]

    def safe_encode(self, text, max_len=None, vocab_size=None):
        """安全编码：自动把越界索引回退到 <UNK>（词表可能小于模型期望）"""
        tokens = self.encode(text, max_len=max_len)
        if vocab_size is None:
            vocab_size = self.next_idx
        return [idx if 0 <= idx < vocab_size else UNK_IDX for idx in tokens]

    # ---------------- 解码 ----------------
    def decode(self, indices):
        """解码索引为文本（跳过全部特殊 token）"""
        chars = []
        special_ids = set(self.SPECIAL_TOKENS.values())
        for idx in indices:
            if idx in special_ids:
                continue
            if idx in self.idx2word:
                chars.append(self.idx2word[idx])
        return "".join(chars)

    # ---------------- 持久化 ----------------
    def save_vocab(self, filepath):
        """保存完整词表（含 BPE 合并规则），保证加载后可精确重现编码"""
        data = {
            "word2idx": self.word2idx,
            "idx2word": {int(k): v for k, v in self.idx2word.items()},
            "next_idx": self.next_idx,
            # 直接保存字符串 pair，避免多字符子词对在 JSON 中被拆成单字符导致规则错位
            "bpe_merges": [[p[0], p[1], m] for p, m in self.bpe_merges],
            "token_freqs": dict(self.token_freqs),
            "vocab_size": self.vocab_size,
            "token_scripts": dict(self.token_scripts),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath):
        """加载完整词表（含 BPE 合并规则），兼容新旧两种存储格式"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
            self.next_idx = data["next_idx"]
            if data.get("vocab_size"):
                self.vocab_size = data["vocab_size"]
            raw_merges = data.get("bpe_merges", [])
            self.bpe_merges = []
            for item in raw_merges:
                if len(item) == 3:

                    def _restore_pair(x):
                        # 旧格式: ['安','晚']（多字符被拆）→ join 还原；新格式: '安晚' 原样
                        return x if isinstance(x, str) else "".join(x)

                    self.bpe_merges.append(((_restore_pair(item[0]), _restore_pair(item[1])), item[2]))
                else:
                    self.bpe_merges.append(((item[0], item[1]), item[2]))
            self.token_freqs = Counter(data.get("token_freqs", {}))
            self.token_scripts = dict(data.get("token_scripts", {}))
            if not self.token_scripts:
                self._ensure_token_scripts()      # 旧词表：按字符补齐语种标签
            return True
        except Exception as e:
            print(f"  [!] 词表加载失败: {e}")
            return False


# 旧版别名兼容
SimpleTokenizer = BPEAnwanTokenizer
EnhancedTokenizer = BPEAnwanTokenizer


def _encode_nopad(tokenizer, text, max_len=None):
    """BPE 编码但不补 <PAD>（多轮历史拼接/记忆注入用，避免 PAD 混入有效序列）"""
    tokens = []
    for word in tokenizer._pre_tokenize(text):
        word_tokens = tokenizer._apply_merges(list(word))
        for t in word_tokens:
            tokens.append(tokenizer.word2idx.get(t, UNK_IDX))
    return tokens[:max_len] if max_len is not None else tokens


def build_multi_turn_src(tokenizer, history, question, max_len=None,
                         history_turns=None, history_max_tokens=None):
    """
    构造带显式多轮历史的 src 序列（P1-1）：
        <BOS> <USER> q1 <BOT> a1 <USER> q2 <BOT> a2 ... <USER> q_now
    - history: [(user_q, bot_reply), ...]（旧→新）
    - 历史部分预算 history_max_tokens（从近到远截断），当前问题始终保留在末尾
    - 返回不超过 max_len 的 token 列表
    """
    max_len = max_len or config.MAX_LEN
    history_turns = history_turns or config.HISTORY_TURNS
    history_max_tokens = history_max_tokens if history_max_tokens is not None else config.HISTORY_MAX_TOKENS

    # 历史段：从最近的轮次往前拼，控制总预算
    hist_tokens = []
    for u, a in reversed(history[-history_turns:] if history_turns else history):
        u_ids = _encode_nopad(tokenizer, u)
        a_ids = _encode_nopad(tokenizer, a)
        seg = [USER_IDX] + u_ids + [BOT_IDX] + a_ids
        if len(hist_tokens) + len(seg) > history_max_tokens:
            break  # 预算不足，丢弃更早的轮次（保留最近的）
        hist_tokens = seg + hist_tokens

    q_ids = _encode_nopad(tokenizer, question)
    seq = [BOS_IDX] + hist_tokens + [USER_IDX] + q_ids
    # 超长截断：保留尾部（当前问题 + 最近历史），BOS 保持位置 0
    if len(seq) > max_len:
        seq = [BOS_IDX] + seq[-(max_len - 1):]
    return seq


def build_src_with_memory(tokenizer, memory_texts, history, question,
                          max_len=None, history_turns=None,
                          history_max_tokens=None, mem_max_tokens=None,
                          top_k=None):
    """
    在历史之前插入外部记忆段落（P1-2）：
        <BOS> [<USER> 记忆片段 <BOT>]... [<USER> q1 <BOT> a1]... <USER> q_now
    - memory_texts: 检索命中并截断后的记忆文本列表（每条再截断 MEMORY_MAX_TOKENS）
    - 无记忆时等价 build_multi_turn_src
    """
    base = build_multi_turn_src(tokenizer, history, question,
                                max_len=None, history_turns=history_turns,
                                history_max_tokens=history_max_tokens)
    mem_seq = []
    cap = config.MEMORY_MAX_TOKENS if mem_max_tokens is None else mem_max_tokens
    for mt in memory_texts[:(top_k if top_k is not None else config.MEMORY_TOP_K)]:
        ids = _encode_nopad(tokenizer, mt, max_len=cap)
        if ids:
            mem_seq += [USER_IDX] + ids + [BOT_IDX]
    if not mem_seq:
        return base
    max_len = max_len or config.MAX_LEN
    seq = [BOS_IDX] + mem_seq + (base[1:] if base and base[0] == BOS_IDX else base)
    if len(seq) > max_len:
        seq = [BOS_IDX] + seq[-(max_len - 1):]
    return seq


def pack_content_segments(tokenizer, texts, max_len=None, max_segs=None):
    """把多段文本打进一条 src：<BOS> t1 <SEP> t2 <SEP> t3 ...
    超出 max_len 时从左截，尽量保住最后一段（当前问题 / 主文本）。"""
    max_len = max_len or config.MAX_LEN
    max_segs = max_segs or getattr(config, "CONTENT_PACK_MAX_SEGS", 4)
    segs = []
    for t in texts[:max_segs]:
        ids = _encode_nopad(tokenizer, t)
        if ids:
            segs.append(ids)
    if not segs:
        return [BOS_IDX]
    bodies = []
    for i, ids in enumerate(segs):
        bodies.append(([] if i == 0 else [SEP_IDX]) + ids)
    tail = []
    for body in reversed(bodies):
        need = len(body) + len(tail)
        if 1 + need > max_len:
            room = max_len - 1 - len(tail)
            if room > 1:
                tail = body[-room:] + tail
            break
        tail = body + tail
    return ([BOS_IDX] + tail)[:max_len]


def _segment_ids_from_src(src, max_segs):
    """从 <SEP>/<USER> 切段号，供内容通道段 embedding 使用。src: [B, T] -> [B, T]"""
    markers = (src == SEP_IDX) | (src == USER_IDX)
    seg = markers.long().cumsum(dim=1)
    return seg.clamp(max=max(int(max_segs) - 1, 0))


def extract_lookup_term(text):
    """从"X是什么意思 / X用英语怎么说 / 翻译成英文的X"这类问句里抽出待查词。
    返回 None 表示抽不出来（让调用方走正常生成路径）。"""
    t = (text or "").strip()
    t = re.sub(r"[?？。！!,.，。]+$", "", t).strip()
    pats = [
        r"^(?:请问|请|帮我|帮忙|查一下|查查|查|解释|介绍|说说)?\s*(.+?)\s*(?:是什么意思|是啥意思|什么意思|啥意思|啥{0,2}字|怎么读|怎么念|念什么|是啥)",
        r"^(?:请问|请|帮我|帮忙)?\s*(.+?)\s*(?:用|用.{0,3}?)(?:英语|英文|日语|日文|韩语|韩文|俄语|法语|德语|西班牙语|意大利语|泰语|阿拉伯语).{0,3}(?:怎么说|怎么讲|怎么读|怎么写|是什么)",
        r"(?:翻译|译)(?:成|为)?.{0,4}?(?:英语|英文|日语|日文|韩语|韩文|俄语|法语|德语|西班牙语|意大利语|泰语|阿拉伯语)\s*[:：]?\s*(.+)",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            term = m.group(1).strip()
            term = re.sub(r"^(?:这个|那个|它|他|她|的)+", "", term).strip()
            if 0 < len(term) <= 24:
                return term
    # 纯生僻字/词直接问："龘" "serendipity"
    m = re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]{1,3}", t)
    if m:
        return t
    m = re.fullmatch(r"[A-Za-z][A-Za-z\-']{1,30}", t)
    if m:
        return t
    return None


class WordBook:
    """
    生词本：查生僻词 / 外语词。用户自己维护，模型不强答。
    ------------------------------------------------------------------
    为什么需要它：从零训练的模型**根本不知道"龘"念 dá、
    意思是什么**。让它自由生成 = 编一个错答案给你，而且语气还特别自信。
    与其这样，不如接一个可维护的词典：查到就答，查不到就老实说
    "我不知道"，并告诉用户怎么往词本里加。
    这是把"不知道"从幻觉变成产品的正确做法。

    存储：data/models/wordbook.json
        {"龘": {"meaning": "龙群飞行貌", "pinyin": "dá", "note": ""},
         "serendipity": {"meaning": "意外发现珍奇事物的运气", "pinyin": "", "note": ""}}
    """

    def __init__(self, filepath=None):
        self.filepath = filepath or config.WORDBOOK_FILE
        self.words = {}
        self.load()

    @staticmethod
    def norm(term):
        return (term or "").strip().lower()

    def load(self):
        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.words = data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"  [!] 生词本加载失败: {e}")
            self.words = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.words, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.filepath)
            return True
        except Exception as e:
            print(f"  [!] 生词本保存失败: {e}")
            return False

    def get(self, term):
        return self.words.get(self.norm(term))

    def put(self, term, meaning, pinyin="", note=""):
        term = (term or "").strip()
        if not term or not (meaning or "").strip():
            return False
        self.words[self.norm(term)] = {"term": term, "meaning": meaning.strip(),
                                       "pinyin": pinyin.strip(), "note": note.strip()}
        return self.save()

    def render(self, term):
        e = self.get(term)
        if not e:
            return None
        head = e.get("term") or term
        if e.get("pinyin"):
            head += f" [{e['pinyin']}]"
        out = f"{head}：{e['meaning']}"
        if e.get("note"):
            out += f"（{e['note']}）"
        return out

    def stats(self):
        return len(self.words)


class ExternalMemory:
    """
    P1-2 外部长期记忆层（零新依赖）：
    - 表示：BPE token 命中直方图（稀疏字典向量，L2 归一化）
    - 分工：模型内部 state 只负责"融合质量"；事实性长记忆放这里落盘持久化，
      对话时 Top-K 检索结果注入 src（与多轮历史并列），久远低活跃记忆平滑遗忘。
    - 持久化：JSON 落盘到 data/models/external_memory.json
    """

    def __init__(self, tokenizer, filepath=None):
        self.tokenizer = tokenizer
        self.filepath = filepath or config.MEMORY_FILE
        self.memories = []    # [{id,text,vec,ts,hits,weight,kind}]
        self.archives = []    # [{id,text,ts,hits,kind}] 平滑遗忘后的软删除区（可恢复）
        self._next_id = 0
        self._dirty = False
        # ---- IDF 统计（记忆集上的文档频率）----
        self._df = Counter()
        self._df_n = 0
        self._df_ratio_at_rebuild = None
        # 概念图谱：跨记忆关联（弹幕/私聊共用，按档位决定是否启用扩散）
        self.graph = MemoryGraph()
        self.load()

    # ---------------- 向量化 ----------------
    # ---------------- IDF 加权（降低记忆间干扰）----------------
    def _rebuild_df(self):
        """重建文档频率表，并按新 IDF 重新计算全部记忆向量。
        IDF 必须在"记忆集"这个语料上统计才成立：像"喵""你""的"这类高频
        特征在所有记忆里都出现，不该主导相似度；原实现按裸词频加权，
        导致 add() 的 0.65 阈值容易把不同主题的记忆误判成同一条。"""
        self._df = Counter()
        for m in self.memories:
            vec = m.get("vec") or {}
            for k in vec:
                self._df[k] += 1
        self._df_n = len(self.memories)
        self._df_ratio_at_rebuild = self._df_n
        # 权重随 df 变化，记忆向量必须一起重算，否则记忆侧与查询侧 IDF 不一致
        for m in self.memories:
            if m.get("text"):
                m["vec"] = self._weight_vec(m["text"])
                self.graph.index(m["id"], m["vec"], df=self._df, n_docs=self._df_n)

    def _maybe_rebuild_idf(self):
        """记忆规模变化超过 IDF_REBUILD_RATIO 倍时重建（增量 df 会让旧记忆
        的 IDF 相对查询侧漂移；定期全量重算把漂移压住，代价可控）。"""
        if self._df_ratio_at_rebuild is None:
            return
        n = len(self.memories)
        if n == 0:
            return
        if n >= self._df_ratio_at_rebuild * config.IDF_REBUILD_RATIO:
            self._rebuild_df()
        elif n < self._df_ratio_at_rebuild * 0.6:   # 遗忘/衰减后也校准一次
            self._rebuild_df()

    def _weight_vec(self, text):
        """文本 → IDF 加权 + L2 归一化的稀疏向量"""
        raw = self._raw_counts(text)
        if not raw:
            return {}
        n = max(self._df_n, 1)
        weighted = {k: v * (math.log((n + 1) / (self._df.get(k, 0) + 1)) + 1.0)
                    for k, v in raw.items()}
        norm = math.sqrt(sum(v * v for v in weighted.values()))
        if norm <= 0:
            return {}
        return {k: round(v / norm, 6) for k, v in weighted.items()}

    def _vectorize(self, text):
        """记忆/查询文本 → L2 归一化稀疏 dict（{feature_id: weight}）。
        特征 = BPE token 命中直方图（剔除特殊 token，UNK 占位不携带语义）
             + 字符 bigram 兜底（无词典依赖，中文短记忆在词表覆盖不足时仍可判相似）。
        bigram 用稳定哈希（zlib.crc32）作 feature id，避开 token id 空间。
        权重在 _weight_vec 里按 IDF 缩放。"""
        return self._weight_vec(text)

    def _raw_counts(self, text):
        """原始特征计数（未加权），供 IDF 加权使用。
        三类特征：
          1) BPE token 命中直方图（剔除特殊 token，UNK 占位不携带语义）
          2) 字符 unigram —— 叠字兜底。"喵喵喵""哈哈哈"这类叠字的 bigram
             全部满足 a==b 被跳过，若没有 unigram 兜底会产生空向量：
             既无法入库去重（add 里 `if vec and ...` 短路），也永远搜不到
             （search 里 `if not qv: return []`）。
          3) 字符 bigram —— 中文短记忆在词表覆盖不足时的主要区分信号。"""
        special_ids = set(self.tokenizer.SPECIAL_TOKENS.values())
        ids = [i for i in _encode_nopad(self.tokenizer, text) if i not in special_ids]
        freq = {}
        for i in ids:
            freq[i] = freq.get(i, 0) + 1
        chars = [ch for ch in text if ch.strip()]
        GRAM_BASE = 1000000
        CHAR_BASE = 2000000
        for ch in chars:                      # 叠字/短文本兜底
            g = CHAR_BASE + zlib.crc32(ch.encode("utf-8")) % GRAM_BASE
            freq[g] = freq.get(g, 0) + 1
        for a, b in zip(chars, chars[1:]):
            if a == b:
                continue                       # 叠字 bigram 不贡献区分度
            g = zlib.crc32((a + b).encode("utf-8")) % GRAM_BASE
            freq[GRAM_BASE + g] = freq.get(GRAM_BASE + g, 0) + 1
        return freq

    @staticmethod
    def _dot(a, b):
        """稀疏字典点积（遍历较小的 dict）"""
        if not a or not b:
            return 0.0
        small, other = (a, b) if len(a) <= len(b) else (b, a)
        return sum(v * other[k] for k, v in small.items() if k in other)

    # ---------------- 增 / 查 / 删 ----------------
    def add(self, text, kind="memory", scope=None):
        """新增记忆；与**同 scope 内**已有记忆相似(≥0.65，词面改写/同义表达判为重记)
        时累加命中次数。返回 id 或 None。
        scope: 记忆归属域（通常是发送者 id / 会话 id）。同 scope 才去重 ——
        公开场合里不同人的记忆是不同的事实，"他喜欢火锅" 不该被合并进
        "她讨厌火锅"，那会直接把一个人的画像污染给另一个人。"""
        text = (text or "").strip()
        if not text:
            return None
        self._maybe_rebuild_idf()          # 规模变化后先校准 IDF
        vec = self._vectorize(text)
        for m in self.memories:
            if m.get("scope") != scope:
                continue
            if vec and self._dot(m["vec"], vec) >= 0.65:
                m["hits"] += 1
                m["ts"] = time.time()
                m["weight"] = min(1.0, m["weight"] + 0.05)   # 复习强化
                self._dirty = True
                self.save()
                return m["id"]
        m = {"id": self._next_id, "text": text, "vec": vec,
             "ts": time.time(), "hits": 1, "weight": 1.0, "kind": kind,
             "scope": scope,
             "script": detect_script(text)}   # 方案A软分区：记忆入库时打语种标签
        self._next_id += 1
        self.memories.append(m)
        self.graph.index(m["id"], vec, df=self._df, n_docs=self._df_n)
        for k in vec:                      # 增量更新文档频率
            self._df[k] += 1
        self._df_n += 1
        self._dirty = True
        self.save()
        return m["id"]

    @staticmethod
    def get_profile(name=None):
        """取检索档位；未知档位名回落到 DEFAULT_PROFILE"""
        return config.RETRIEVAL_PROFILES.get(
            name or config.DEFAULT_PROFILE, config.RETRIEVAL_PROFILES[config.DEFAULT_PROFILE])

    def search(self, query, top_k=None, min_score=None, touch=True,
               scope=None, profile=None, spread=None):
        """
        两阶段检索：向量粗筛 → 概念图谱一跳激活扩散 → 重排。
        - scope: 只在该归属域内检索（弹幕场景下的隔离硬需求；None=全域，
                 供 CLI 手动查全部）
        - profile: "barrage" | "dm"，决定 top_k / min_score / 是否扩散
        返回 [(score, text, id), ...]（降序）
        """
        prof = self.get_profile(profile)
        if top_k is None:
            top_k = prof["top_k"]
        if min_score is None:
            min_score = prof["min_score"]
        if spread is None:
            spread = prof["spread"]
        self._maybe_rebuild_idf()
        qv = self._vectorize(query)
        if not qv or not self.memories:
            return []

        # ---- ANWAN_RETRIEVAL_MODE=facts_only：跳过向量粗筛，只走 MemoryGraph + scope 过滤 ----
        # 不改 ExternalMemory 对外接口，只在 search 内部按开关分支
        retrieval_mode = getattr(config, "ANWAN_RETRIEVAL_MODE", "full")
        if retrieval_mode == "facts_only":
            # facts_only 模式：只返回 kind="fact"，跳过向量粗筛
            seeds = {}
            for m in self.memories:
                if m.get("kind") != "fact":
                    continue
                if scope is not None and m.get("scope") != scope:
                    continue
                # 不使用向量相似度，用权重+时间衰减排序
                score = m.get("weight", 0.5)
                if score >= min_score:
                    seeds[m["id"]] = score
        else:
            # full 模式：正常向量粗筛
            q_script = detect_script(query)       # 方案A：查询的语种归属
            bonus = getattr(config, "SEARCH_LANG_BONUS", 0.15)
            seeds = {}
            for m in self.memories:
                if not m["vec"]:
                    continue
                if scope is not None and m.get("scope") != scope:
                    continue                      # 隔离：不跨归属域取记忆
                score = self._dot(qv, m["vec"]) * m["weight"]
                if m.get("script") == q_script:
                    score = score + bonus         # 同语种优先，压跨语种噪声
                if score >= min_score:
                    seeds[m["id"]] = score
        if not seeds:
            return []
        if touch and seeds:
            for m in self.memories:
                if m["id"] in seeds:
                    m["ts"] = time.time()
                    m["hits"] += 1
            self._dirty = True
            self.save()

        # ---- 阶段二：图谱一跳激活扩散（私聊档位才开）----
        # 扩散能带出"向量不像但概念相连"的记忆（跨记忆关联）；
        # 弹幕档位关闭，因为那里 1-3 秒的预算换不来关联召回的收益。
        if spread and seeds:
            top_seed = sorted(seeds.items(), key=lambda kv: -kv[1])[:top_k]
            self.graph.co_retrieved([mid for mid, _ in top_seed])   # 共现即连边
            boosted = self.graph.spread(dict(top_seed))
            # 扩散进来的记忆原始分可能远低于 min_score（这正是"关联召回"的定义），
            # 但不能无限兜底：加一道地板，只接受达到 min_score×比例的 boost，
            # 否则一次强命中就能把八竿子打不着的邻居全拖进回复里。
            floor = min_score * config.GRAPH_SPREAD_MIN_RATIO
            allowed = {m["id"] for m in self.memories
                       if scope is None or m.get("scope") == scope}
            for mid, sc in boosted.items():
                if mid in seeds:
                    continue                      # 已经是向量直命中，不受地板约束
                # 隔离二次校验：扩散出来的邻居必须仍在同 scope 内。
                # 图谱边是跨记忆共享概念建立的，天然会横跨不同 scope；
                # 不在这里挡一道，弹幕场景就会把别人的记忆拖进当前回复。
                if scope is not None and mid not in allowed:
                    continue
                if sc >= floor:
                    seeds[mid] = sc

        id2mem = {m["id"]: m for m in self.memories}
        scored = [(round(sc, 4), id2mem[mid]["text"], mid)
                  for mid, sc in seeds.items() if mid in id2mem]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def forget(self, key=None):
        """软删除：key=None 清空全部；int=按 id；str=按文本子串。删除项移入归档（可恢复）"""
        if key is None:
            moved, self.memories = self.memories, []
            self._next_id = 0
        else:
            moved, keep = [], []
            for m in self.memories:
                hit = (m["id"] == key) if isinstance(key, int) else (key in m["text"])
                (moved if hit else keep).append(m)
            self.memories = keep
        if moved:
            self.archives = [dict(m, vec=None) for m in moved] + self.archives
            if len(self.archives) > config.MEMORY_MAX_ARCHIVE:
                self.archives = self.archives[:config.MEMORY_MAX_ARCHIVE]
        for mm in moved:
            self.graph.unindex(mm["id"])
        self._df_ratio_at_rebuild = None    # 记忆集已变，标记 IDF 待重建
        self._dirty = True
        self.save()
        return len(moved)

    def restore(self, index=0):
        """从归档恢复一条记忆（默认最近归档）"""
        if not self.archives:
            return None
        m = self.archives.pop(index)
        m["weight"] = 0.8
        m["hits"] = max(m.get("hits", 1), 1)
        self._next_id = max(self._next_id, m["id"] + 1)
        if any(x["id"] == m["id"] for x in self.memories):
            m["id"] = self._next_id
            self._next_id += 1
        self.memories.append(m)
        self.graph.index(m["id"], m.get("vec"), df=self._df, n_docs=self._df_n)
        self._df_ratio_at_rebuild = None    # 恢复的记忆 vec 是归档态，需重建
        self._dirty = True
        self.save()
        return m["text"]

    def decay(self):
        """平滑遗忘：全体活跃记忆权重 × MEMORY_DECAY；低权重且低活跃/久未访问的转归档"""
        now = time.time()
        moved, keep = [], []
        for m in self.memories:
            m["weight"] *= config.MEMORY_DECAY
            idle_days = (now - m["ts"]) / 86400.0
            if m["weight"] < config.MEMORY_EXPIRE_WEIGHT and (m["hits"] < 2 or idle_days > 30):
                moved.append(m)
            else:
                keep.append(m)
        self.memories = keep
        if moved:
            self.archives = [dict(m, vec=None) for m in moved] + self.archives
            if len(self.archives) > config.MEMORY_MAX_ARCHIVE:
                self.archives = self.archives[:config.MEMORY_MAX_ARCHIVE]
        for mm in moved:
            self.graph.unindex(mm["id"])
        for mm in moved:
            self.graph.unindex(mm["id"])
        self._df_ratio_at_rebuild = None    # 记忆集已变，标记 IDF 待重建
        self._dirty = True
        self.save()
        return len(moved), len(self.memories), len(self.archives)

    # ---------------- 持久化 ----------------
    def save(self):
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            data = {
                "version": 2,
                "next_id": self._next_id,
                "memories": [
                    {k: m.get(k) for k in ("id", "text", "vec", "ts", "hits", "weight", "kind", "scope")}
                    for m in self.memories
                ],
                "archives": [
                    {k: m.get(k) for k in ("id", "text", "ts", "hits", "kind", "scope")}
                    for m in self.archives
                ],
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"  [!] 外部记忆保存失败: {e}")
        self._dirty = False

    def load(self):
        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.memories = data.get("memories", [])
            self.archives = data.get("archives", [])
            self._next_id = data.get("next_id", 0)
            if not self.memories and self.archives:
                self._next_id = max([m["id"] for m in self.archives], default=0) + 1
            for m in self.memories:
                m.setdefault("weight", 1.0)
                m.setdefault("hits", 1)
                m.setdefault("kind", "memory")
                m.setdefault("vec", {})
                m.setdefault("scope", None)
            for m in self.archives:
                m.setdefault("hits", 1)
                m.setdefault("kind", "memory")
                m.setdefault("scope", None)
            # 词表可能已变（重训后），旧 vec 与新特征空间对不上 → 整体重算
            self._rebuild_df()
            self._dirty = False
        except Exception as e:
            print(f"  [!] 外部记忆加载失败: {e}")

    def stats(self):
        total_vec = sum(1 for m in self.memories if m.get("vec"))
        return {"active": len(self.memories), "archived": len(self.archives),
                "with_vec": total_vec, "graph": self.graph.stats()}

    def list(self, limit=10):
        """最近 limit 条活跃记忆（供 CLI 展示）"""
        items = sorted(self.memories, key=lambda m: m["ts"], reverse=True)
        return [(m["id"], m["text"], round(m["weight"], 3), m["hits"]) for m in items[:limit]]

    # ---------------- 冲突检测与仲裁 ----------------
    CONFLICT_PAIRS = [
        ("喜欢", "讨厌"), ("喜欢", "不喜欢"), ("喜欢", "恨"), ("爱", "恨"), ("爱", "不爱"),
        ("是", "不是"), ("是", "否"), ("是", "非"), ("对", "错"), ("真", "假"),
        ("会", "不会"), ("会", "不能"), ("能", "不能"), ("能", "无法"), ("可以", "不可以"),
        ("有", "没有"), ("有", "无"), ("存在", "不存在"),
        ("去", "不去"), ("去", "没去"), ("来", "不来"),
        ("买", "不买"), ("吃", "不吃"), ("要", "不要"), ("做", "不做"),
        ("开心", "难过"), ("开心", "伤心"), ("开心", "生气"), ("高兴", "难过"),
        ("好", "不好"), ("好", "坏"), ("大", "小"), ("多", "少"),
        ("热", "冷"), ("高", "低"), ("快", "慢"), ("新", "旧"), ("强", "弱"),
    ]

    def conflict_detect(self, scored_memories):
        """
        检测检索结果中的记忆冲突。
        输入: search 返回的 [(score, text, id), ...]
        返回: 冲突组列表，每组是 [memory_dict, ...]（同一主题但语义矛盾）
        检测策略：
          1) 向量相似度 ≥0.75 视为同一主题（同语义空间）
          2) 同主题内检查互斥关键词对（如"喜欢"vs"讨厌"）
        """
        if len(scored_memories) < 2:
            return []
        # id -> memory_dict 映射
        id2mem = {m["id"]: m for m in self.memories}
        mems = [id2mem.get(mid) for _s, _t, mid in scored_memories]
        mems = [m for m in mems if m and m.get("vec")]
        n = len(mems)
        if n < 2:
            return []
        # 按向量相似度做并查集分组（高相似=同主题）
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._dot(mems[i]["vec"], mems[j]["vec"])
                if sim >= 0.75:
                    union(i, j)
        groups = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(mems[idx])
        # 对每组检查互斥关键词
        conflict_groups = []
        for group in groups.values():
            if len(group) < 2:
                continue
            texts = [m["text"] for m in group]
            has_conflict = False
            for pos, neg in self.CONFLICT_PAIRS:
                pos_in = [pos in t for t in texts]
                neg_in = [neg in t for t in texts]
                if any(pos_in) and any(neg_in):
                    # 存在至少一条含正向词、至少一条含负向词 → 冲突
                    has_conflict = True
                    break
            if has_conflict:
                conflict_groups.append(group)
        return conflict_groups

    def conflict_arbitrate(self, conflict_groups):
        """
        对冲突组进行仲裁，每组只保留最优记忆。
        仲裁优先级（综合得分降序）：
          score = weight × hits × (1 + 0.1 × recency_score) / (age_days + 1)
          recency_score = 当天+1，7天内+0.5，30天内+0.2， older+0
        返回: (保留的 id 列表, 被剔除的 id 列表)
        """
        now = time.time()
        keep_ids, drop_ids = [], []
        for group in conflict_groups:
            scored = []
            for m in group:
                age_days = max(0.0, (now - m["ts"]) / 86400.0)
                if age_days < 1:
                    recency = 1.0
                elif age_days < 7:
                    recency = 0.5
                elif age_days < 30:
                    recency = 0.2
                else:
                    recency = 0.0
                score = m["weight"] * m["hits"] * (1 + 0.1 * recency) / (age_days + 1)
                scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            keep_ids.append(scored[0][1]["id"])
            drop_ids.extend(m["id"] for _s, m in scored[1:])
        return keep_ids, drop_ids


class MemoryGraph:
    """
    概念图谱：节点=记忆，边=概念共现强度。
    ------------------------------------------------------------------
    为什么需要它：ExternalMemory 的每条记忆是孤岛，只能做"这条像不像那句"
    的独立比对，答不了"他喜欢什么""上次那个项目后来怎么样了"这类跨记忆关联。
    NachoBot/MaiBot 的 Hippocampus 用的就是概念图 + 激活扩散，但它的检索要
    靠 ReAct Agent 反复调 LLM（官方推荐 15-25 秒）。这里用零依赖方式复刻
    同样的效果：把每条记忆 IDF 权重最高的 N 个特征当作它的"概念集"，两条
    记忆概念有交集就连边，检索时在向量召回的候选集内做一跳扩散。

    概念不额外提取：直接复用 _weight_vec 已算好的 IDF 加权向量。IDF 天然
    会把"火锅""毛毛"这类区分性词顶上来、把"我""的"这类虚词压下去，所以
    top-N 特征就是最好的概念集 —— 零新依赖，也不必引入 jieba。
    """

    def __init__(self, top_concepts=None, decay=None):
        self.concepts = {}     # mem_id -> frozenset(概念 feature_id)
        self.adj = {}          # mem_id -> {邻居 mem_id: 强度}（O(度数) 扩散）
        self.by_concept = {}   # 概念 -> {mem_id}（写入时反查邻居，避免 O(n²)）
        self._df = {}          # 特征 -> 文档频率（判断概念是否有区分度）
        self.n_docs = 0
        self.top_concepts = top_concepts or config.GRAPH_TOP_CONCEPTS
        self.decay = decay if decay is not None else config.GRAPH_SPREAD_DECAY

    def _qualifies(self, feature_id):
        """概念资格：只认有区分度的特征。
        记忆集小的时候（比如只有 5 条），几乎所有特征的 df 都等于 1，IDF
        全部并列，"两""了"这种字会被当成概念，把毫不相干的记忆连起来。
        这里要求该特征只出现在不超过 max(2, GRAPH_CONCEPT_MAX_DF) 的记忆里。"""
        if self.n_docs <= 2:
            return True
        limit = max(2, int(self.n_docs * config.GRAPH_CONCEPT_MAX_DF))
        return self._df.get(feature_id, 0) <= limit

    def _bump(self, a, b):
        strength = self.adj.get(a, {}).get(b, 0) + 1
        self.adj.setdefault(a, {})[b] = strength
        self.adj.setdefault(b, {})[a] = strength

    def index(self, mem_id, vec, df=None, n_docs=None):
        """提取概念集（权重最高**且有区分度**的 N 个特征），
        并立即与已有同概念记忆建立**弱边**（强度 1）。
        边必须在写入时建：若只在"被同一查询同时召回"时才连边，冷启动阶段
        每次检索往往只召回 1 条，Hebbian 学习根本起不来。"""
        if df is not None:
            self._df = df
        if n_docs is not None:
            self.n_docs = n_docs
        if not vec:
            self.unindex(mem_id)
            return
        # 概念集 = 全部"有区分度"的特征（按权重排序，最多保留 top_concepts 个）。
        # 注意不能用"权重最高的 N 个"当概念：那 N 个最稀有的特征按定义就是
        # 这条记忆独有的，两条同主题记忆反而连不上（"毛毛胖了"和"医生建议
        # 控制饮食"共享的是 医/它/要 这些中等稀有度特征）。链接靠的是
        # "共享任一有区分度的特征"，不是 top-N 相等。
        # 排序方向很关键：按 df 降序（越"被多条记忆共享"越靠前），而不是按
        # IDF 权重降序。权重最高的那些特征按定义就是本条独有的（df≈0），
        # 留它们当概念会导致同主题记忆连不上；真正把"毛毛胖了"和"医生建议
        # 控制饮食"连起来的是 医/它/要 这类 df 中等的共享特征。
        cand = [(k, self._df.get(k, 0), w) for k, w in vec.items() if self._qualifies(k)]
        if not cand:                      # 记忆集太小、IDF 无区分度 → 退回原始 top-N
            cand = [(k, 0, w) for k, w in vec.items()]
        cand.sort(key=lambda t: (-t[1], -t[2]))
        picked = [k for k, _df, _w in cand[:self.top_concepts]]
        self.unindex(mem_id)
        concepts = frozenset(picked)
        self.concepts[mem_id] = concepts
        for c in concepts:
            self.by_concept.setdefault(c, set()).add(mem_id)
        for c in concepts:                # 写入即建弱边
            for other in self.by_concept.get(c, ()):
                if other != mem_id:
                    self._bump(other, mem_id)

    def unindex(self, mem_id):
        """移除节点及其全部边（遗忘/归档时调用，防止图谱指向不存在的记忆）"""
        for c in self.concepts.pop(mem_id, ()):
            holders = self.by_concept.get(c)
            if holders is not None:
                holders.discard(mem_id)
                if not holders:
                    self.by_concept.pop(c, None)
        for nb in list(self.adj.get(mem_id, {})):
            self.adj[nb].pop(mem_id, None)
            if not self.adj[nb]:
                self.adj.pop(nb, None)
        self.adj.pop(mem_id, None)

    def connect(self, a, b):
        """概念有交集才加强关联（强度 +1，strength/(strength+1) 饱和）"""
        if a == b:
            return
        ca, cb = self.concepts.get(a), self.concepts.get(b)
        if not ca or not cb or not (ca & cb):
            return
        self._bump(a, b)

    def co_retrieved(self, ids):
        """同一次检索里共同命中的记忆两两连边 —— 边强度 = 共现次数"""
        ids = list(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.connect(ids[i], ids[j])

    def spread(self, seeds):
        """一跳激活扩散：种子分数沿边衰减传递给邻居。
        seeds: {mem_id: 原始分数} → 返回增强后的 {mem_id: 分数}（含未命中的邻居）"""
        out = dict(seeds)
        for mid, s in seeds.items():
            for nb, strength in self.adj.get(mid, {}).items():
                boost = s * self.decay * (strength / (strength + 1.0))
                if boost > out.get(nb, 0.0):
                    out[nb] = boost
        return out

    def stats(self):
        return {"nodes": len(self.concepts), "edges": sum(len(v) for v in self.adj.values()) // 2}


# ==================== AWIL 模型内部语言（AnWan Internal Language）====================
# 紧凑机器码协议：定长4字节头部(src/dst/op/flags) + 变长payload
# 设计目标：模拟大模型内部模块通信，旁路日志/调试/监控用

import struct as _struct


class AWIL:
    """
    AWIL 消息帧：
        [src:1B][dst:1B][op:1B][flags:1B][payload_len:2B][payload:N B]
    总头部 6 字节，payload 最长 65535 字节。
    """

    # ---- 模块ID ----
    MOD_BUS = 0x00       # 总线/广播
    MOD_LSTM_A = 0x01    # 内容通道（双向专家 Transformer / 回退 LSTM-A）
    MOD_LSTM_B = 0x02    # 情感通道（LSTM-B）
    MOD_TF1 = 0x03       # 检索（TF-1）
    MOD_TF2 = 0x04       # 生成（TF-2）
    MOD_ROUTER = 0x05    # 中央路由
    MOD_INTENT = 0x06    # 意图头
    MOD_MEMORY = 0x07    # 外部记忆
    MOD_USER = 0x10      # 外部输入
    MOD_BOT = 0x11       # 外部输出

    # ---- 操作码 ----
    OP_ROUTE = 0x01       # 路由决策
    OP_EMOTION = 0x02     # 情感状态传递
    OP_CONTENT = 0x03     # 内容摘要
    OP_RETRIEVE = 0x04    # 检索结果
    OP_GENERATE = 0x05    # 生成单步 token
    OP_INTENT = 0x06      # 意图识别
    OP_MEM_READ = 0x07    # 记忆读取
    OP_MEM_WRITE = 0x08   # 记忆写入
    OP_SYNC = 0x09        # 状态同步

    _MOD_NAMES = {
        MOD_BUS: "BUS", MOD_LSTM_A: "LSTM-A", MOD_LSTM_B: "LSTM-B",
        MOD_TF1: "TF-1", MOD_TF2: "TF-2", MOD_ROUTER: "ROUTER",
        MOD_INTENT: "INTENT", MOD_MEMORY: "MEMORY",
        MOD_USER: "USER", MOD_BOT: "BOT",
    }
    _OP_NAMES = {
        OP_ROUTE: "ROUTE", OP_EMOTION: "EMOTION", OP_CONTENT: "CONTENT",
        OP_RETRIEVE: "RETRIEVE", OP_GENERATE: "GENERATE", OP_INTENT: "INTENT",
        OP_MEM_READ: "MEM_READ", OP_MEM_WRITE: "MEM_WRITE", OP_SYNC: "SYNC",
    }

    def __init__(self, src=0x00, dst=0x00, op=0x00, flags=0, payload=None):
        self.src = int(src) & 0xFF
        self.dst = int(dst) & 0xFF
        self.op = int(op) & 0xFF
        self.flags = int(flags) & 0xFF
        self.payload = payload if payload is not None else b""
        if isinstance(self.payload, str):
            self.payload = self.payload.encode("utf-8")

    def encode(self):
        """→ bytes（紧凑二进制）"""
        pl = self.payload if isinstance(self.payload, bytes) else str(self.payload).encode("utf-8")
        return _struct.pack(">BBBBH", self.src, self.dst, self.op, self.flags, len(pl)) + pl

    @classmethod
    def decode(cls, data):
        """bytes → AWIL"""
        if len(data) < 6:
            return None
        src, dst, op, flags, plen = _struct.unpack(">BBBBH", data[:6])
        pl = data[6:6 + plen]
        return cls(src, dst, op, flags, pl)

    @classmethod
    def from_floats(cls, src, dst, op, floats, flags=0):
        """payload 为 float32 数组（用于传递张量摘要）"""
        payload = _struct.pack(">" + "f" * len(floats), *floats)
        return cls(src, dst, op, flags, payload)

    def to_floats(self):
        """把 payload 当 float32 数组解码"""
        pl = self.payload
        n = len(pl) // 4
        return list(_struct.unpack(">" + "f" * n, pl[: n * 4]))

    @classmethod
    def from_bytes_vals(cls, src, dst, op, vals, flags=0):
        """payload 为 uint8 数组"""
        payload = bytes(int(v) & 0xFF for v in vals)
        return cls(src, dst, op, flags, payload)

    # ---- 超短码文本层（10–20 字符，比自然语言省 90%+ token）----
    _TINY_MOD = {
        0x00: "X", 0x01: "A", 0x02: "B", 0x03: "T", 0x04: "F",
        0x05: "R", 0x06: "I", 0x07: "M", 0x10: "U", 0x11: "O",
    }
    _TINY_OP = {
        0x01: "r", 0x02: "e", 0x03: "c", 0x04: "v", 0x05: "g",
        0x06: "i", 0x07: "m", 0x08: "w", 0x09: "s",
    }
    _REV_MOD = {v: k for k, v in _TINY_MOD.items()}
    _REV_OP = {v: k for k, v in _TINY_OP.items()}

    def to_tiny(self):
        """
        超短码文本表示（调试用 / 嵌入上下文用）：
          {src}{dst}{op}[payload 摘要]
        例：XRr[0.5,0.3,0.1,0.1]  →  ROUTER→BUS 路由权重
            BBe[0.2,-0.1]         →  LSTM-B→BUS 情感摘要
            TFg[42,0.85]          →  TF-2→BUS 生成 token#42
        单条通常 10–20 字符，比自然语言指令省 90%+ token。
        """
        sm = self._TINY_MOD.get(self.src, "?")
        dm = self._TINY_MOD.get(self.dst, "?")
        om = self._TINY_OP.get(self.op, "?")
        # payload 摘要：float 取前 4 个，精度 2 位；bytes 取 hex 前 4 字节
        if self.op in (self.OP_ROUTE, self.OP_EMOTION, self.OP_INTENT, self.OP_GENERATE):
            vals = self.to_floats()[:4]
            body = ",".join(f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}" for v in vals)
        else:
            body = self.payload[:4].hex()
        return f"{sm}{dm}{om}[{body}]"

    @classmethod
    def from_tiny(cls, s):
        """超短码字符串 → AWIL（摘要层，payload 为近似值，仅供可读）"""
        if len(s) < 4 or s[3] != "[":
            return None
        src = cls._REV_MOD.get(s[0], 0x00)
        dst = cls._REV_MOD.get(s[1], 0x00)
        op = cls._REV_OP.get(s[2], 0x00)
        body = s[4:].rstrip("]")
        if op in (cls.OP_ROUTE, cls.OP_EMOTION, cls.OP_INTENT, cls.OP_GENERATE):
            try:
                floats = [float(x) for x in body.split(",") if x]
                return cls.from_floats(src, dst, op, floats)
            except Exception:
                return cls(src, dst, op, 0, body.encode("utf-8"))
        else:
            return cls(src, dst, op, 0, body.encode("utf-8"))

    def disasm(self):
        """人类可读的反汇编行（调试用，较长；to_tiny 用于紧凑场景）"""
        sm = self._MOD_NAMES.get(self.src, f"0x{self.src:02X}")
        dm = self._MOD_NAMES.get(self.dst, f"0x{self.dst:02X}")
        om = self._OP_NAMES.get(self.op, f"0x{self.op:02X}")
        pl_str = self.payload[:32].hex() + ("..." if len(self.payload) > 32 else "")
        return f"[{sm:6s} → {dm:6s} | {om:10s} | fl=0x{self.flags:02X} | pl={len(self.payload):4d}B | {pl_str}]"

    def __repr__(self):
        return self.to_tiny()  # 默认用超短码，紧凑优先


class AWILBus:
    """
    AWIL 消息总线（零新依赖，纯 Python）：
    - 环形缓冲（默认容量 1024）
    - 支持按 src/dst/op 过滤读取
    - 自动丢旧消息防内存膨胀
    """

    def __init__(self, capacity=1024):
        self._buf = []
        self._cap = max(capacity, 64)
        self._seq = 0
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, v):
        self._enabled = bool(v)

    def emit(self, msg: AWIL):
        if not self._enabled:
            return
        self._seq += 1
        self._buf.append((self._seq, msg))
        if len(self._buf) > self._cap:
            self._buf.pop(0)

    def emit_route(self, src_mod, dst_mod, weights, entropy=None):
        """发射路由决策消息（weights 为 float 列表）"""
        floats = list(weights) + ([float(entropy)] if entropy is not None else [])
        self.emit(AWIL.from_floats(src_mod, dst_mod, AWIL.OP_ROUTE, floats))

    def emit_emotion(self, src_mod, dst_mod, emotion_vec):
        self.emit(AWIL.from_floats(src_mod, dst_mod, AWIL.OP_EMOTION, list(emotion_vec)))

    def emit_retrieve(self, src_mod, dst_mod, memory_ids):
        """发射检索结果消息（memory_ids 为 int 列表）"""
        self.emit(AWIL.from_bytes_vals(src_mod, dst_mod, AWIL.OP_RETRIEVE, memory_ids))

    def emit_generate(self, src_mod, dst_mod, token_id, prob=None):
        floats = [float(token_id)]
        if prob is not None:
            floats.append(float(prob))
        self.emit(AWIL.from_floats(src_mod, dst_mod, AWIL.OP_GENERATE, floats))

    def emit_intent(self, src_mod, dst_mod, intent_id, confidence):
        self.emit(AWIL.from_floats(src_mod, dst_mod, AWIL.OP_INTENT, [float(intent_id), float(confidence)]))

    def emit_mem(self, src_mod, dst_mod, op, memory_text):
        self.emit(AWIL(src_mod, dst_mod, op, 0, memory_text))

    def dump(self, filter_src=None, filter_dst=None, filter_op=None, last_n=50):
        """返回人类可读的消息流水账（调试用）"""
        lines = []
        for seq, msg in self._buf[-last_n:]:
            if filter_src is not None and msg.src != filter_src:
                continue
            if filter_dst is not None and msg.dst != filter_dst:
                continue
            if filter_op is not None and msg.op != filter_op:
                continue
            lines.append(f"  [{seq:4d}] {msg.to_tiny()}")
        return "\n".join(lines) if lines else "  (无消息)"

    def clear(self):
        self._buf.clear()
        self._seq = 0

    def __len__(self):
        return len(self._buf)


# ==================== 模型（原 anwan/model.py） ====================

def _build_causal_mask(size, device):
    """生成 [size, size] 因果掩码（下三角可见；bool：True=屏蔽未来位置）。
    bool 掩码与 AMP(fp16) 注意力分数兼容，避免 float(-inf) 掩码引发 dtype 冲突。"""
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def _pad_mask(ids, device):
    """[B, T] -> [B, T] bool：True 表示该位置是 padding（被忽略）"""
    return ids == PAD_IDX


def _effective_lengths(ids):
    """每行有效 token 数（BOS 在位置 0，左对齐，PAD 仅在尾部；至少为 1）"""
    return (ids != PAD_IDX).sum(dim=1).clamp(min=1).cpu()


def _packed_lstm(rnn, emb, lengths, hx=None, total_length=None):
    """
    只对有效 token 演化 LSTM 状态，避免 PAD 步把状态冲刷到固定点。
    - emb:      [B, T, E]（含 PAD 行，pack 时按 lengths 截掉）
    - lengths:  [B] 每行有效长度
    - hx:       (h0, c0) 可选（跨轮情感 state）；非 None 且 B>1 时按 sorted_indices 重排
    返回 (output[B,T,E] 尾部补零, hx_out 已还原 batch 序)
    """
    packed = pack_padded_sequence(
        emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
    )
    if hx is not None:
        si = packed.sorted_indices.to(emb.device)
        # AMP 兼容：autocast 下 emb 可能已是 fp16，hx 必须对齐到相同 dtype，否则 cuDNN 报错
        hx = tuple(h[:, si].to(emb.dtype) for h in hx)
        out, hx_out = rnn(packed, hx)
        ui = packed.unsorted_indices.to(emb.device)
        hx_out = tuple(h[:, ui].to(emb.dtype) for h in hx_out)
    else:
        out, hx_out = rnn(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=total_length)
    return out, hx_out


# ==================== MoE 轻量专家模块（decoder 专用） ====================

class _Expert(nn.Module):
    """单个 MoE 专家：小型 FFN（d_model -> expert_ff -> d_model）。"""
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class _MoELayer(nn.Module):
    """轻量 MoE 层：Top-k 路由，每 token 只激活 k 个专家。

    与 nn.TransformerDecoderLayer 配合：替换其 FFN 子层，保留
    多头自注意力 + 交叉注意力不变。aux_loss 在训练时返回供 compute_loss
    累加，推理时不参与梯度。
    """
    def __init__(self, d_model, num_experts=4, top_k=2, expert_ff_dim=2048,
                 dropout=0.1, aux_loss_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        # 路由门
        self.gate = nn.Linear(d_model, num_experts)
        # 专家池
        self.experts = nn.ModuleList([
            _Expert(d_model, expert_ff_dim, dropout) for _ in range(num_experts)
        ])
        self.last_usage = None   # [E] 本 batch 平均路由概率（专家化体检用）

    def expert_param_count(self):
        return sum(p.numel() for e in self.experts for p in e.parameters())

    def forward(self, x):
        """x: [B, T, d_model]；返回 (out, aux_loss)。"""
        B, T, D = x.size()
        x_flat = x.view(B * T, D)
        # 路由分数
        gate_logits = self.gate(x_flat)                         # [B*T, E]
        gate_probs = F.softmax(gate_logits, dim=-1)               # [B*T, E]
        # Top-k + 归一化
        topk_w, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # [B*T, k]
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
        # 按专家聚合输入
        out = torch.zeros_like(x_flat)                            # [B*T, D]
        for e_id in range(self.num_experts):
            mask = (topk_idx == e_id)                              # [B*T, k]
            if not mask.any():
                continue
            tok_idx, k_slot = torch.where(mask)                    # 1-D 索引
            expert_in = x_flat[tok_idx]                            # [N, D]
            expert_w = topk_w[tok_idx, k_slot].unsqueeze(-1)       # [N, 1]
            expert_out = self.experts[e_id](expert_in)             # [N, D]
            out.index_add_(0, tok_idx, expert_w * expert_out)
        out = out.view(B, T, D)
        # 负载均衡辅助 loss（鼓励均匀路由）
        router_prob = gate_probs.mean(dim=0)                       # [E]
        self.last_usage = router_prob.detach()                # 留在设备上，避免每步 GPU 同步
        aux_loss = self.num_experts * (router_prob ** 2).sum() - 1.0
        aux_loss = aux_loss.clamp_min(0.0) * self.aux_loss_weight
        return out, aux_loss


class _MoEDecoderLayer(nn.Module):
    """手写 MoE Decoder 层：保留 MultiheadAttention + CrossAttention，
    FFN 替换为 _MoELayer Top-k 专家池（轻量）。"""
    def __init__(self, d_model, nhead, num_experts=4, top_k=2,
                 expert_ff_dim=2048, dropout=0.1):
        super().__init__()
        # --- 自注意力 ---
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # --- 交叉注意力 ---
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # --- MoE FFN ---
        self.moe_ffn = _MoELayer(d_model, num_experts, top_k, expert_ff_dim, dropout,
                                 aux_loss_weight=getattr(config, "MOE_LOAD_BALANCE_WEIGHT", 0.01))
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # 自注意力
        sa_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(sa_out))
        # 交叉注意力
        ca_out, _ = self.cross_attn(tgt, memory, memory,
                                     key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(ca_out))
        # MoE FFN
        moe_out, aux_loss = self.moe_ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(moe_out))
        return tgt, aux_loss


# ==================== v3 半专家化模块（全域专家 + 情感半专家）====================
# 设计口径（写死在这里，免得日后代码与注释各说各话）：
#   专家模型（纯 MoE）= 没有常驻主干，每条序列 / 每个 token 只激活 Top-k 个专家。
#                       用在内容通道双向 Transformer、检索编码器 TF-1、生成解码器 TF-2。
#   半专家 + 完整模型   = 一个始终激活的「完整模型」主干 + Top-k 个小得多的半专家，
#                       半专家只在主干输出上做门控限幅的小幅修正。
#                       只用在情感通道 LSTM-B —— 因为情感是唯一跨轮持久化的状态，
#                       一旦被稀疏路由「断电」，跨轮人设与 PAD 向量方向会直接漂移。
def _topk_route(probs, top_k):
    """Top-k 硬路由 + 权重归一化。probs: [N, E] -> (w [N, k], idx [N, k])。"""
    topk_w, topk_idx = torch.topk(probs, top_k, dim=-1)
    topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
    return topk_w, topk_idx


def _load_balance_aux(gate_probs, num_experts, aux_weight, min_load=0.0, min_load_weight=0.5):
    """负载均衡辅助 loss（GShard/Switch 风格）：路由越均匀越小。
    必须从 gate_probs（软概率）求导 —— top-k 是硬选择，router 拿不到主 loss 的梯度，
    辅助项是 router 唯一的训练信号来源，去掉它必然坍缩到单专家。
    min_load 是**饥饿下界**：只有均衡项（二次型）时，小 batch / 短训练下某个专家会
    整程没被路由过，其零初始化投影永远停在 0（实测复现：out_projs[0] abs.sum=0.0），
    所以再加一条 hinge —— 平均负载掉到下界以下直接罚，把序列硬推给闲置专家。
    """
    router_prob = gate_probs.mean(dim=0)
    aux = (num_experts * (router_prob ** 2).sum() - 1.0).clamp_min(0.0) * aux_weight
    if min_load > 0:
        aux = aux + (min_load - router_prob).clamp_min(0.0).sum() * min_load_weight
    return aux



# ==================== SSM 主干（Mamba2 / 回退 S6） ====================
# 注意：SSM 回退实现精度低于 Mamba2 原生，文件头已注明。
# 比较报告中会单独标注当前用的是哪种。

_HAS_MAMBA_SSM = False
_Mamba2Native = None
try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2Native
    _HAS_MAMBA_SSM = True
except ImportError:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as _Mamba2Native
        _HAS_MAMBA_SSM = True
    except ImportError:
        _HAS_MAMBA_SSM = False
        _Mamba2Native = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm as _RMSNorm
except ImportError:
    try:
        from mamba_ssm.ops.triton.rms_norm import RMSNorm as _RMSNorm
    except ImportError:
        class _RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-5):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
                return self.weight * (x / rms)


class SequenceBackbone(nn.Module):
    """统一 SequenceBackbone 协议：forward(x, lengths, state=None) -> (out, new_state)；
    state_to_dict()/state_from_dict() 用于跨轮持久化。"""
    def state_to_dict(self):
        return None
    def state_from_dict(self, state_dict):
        pass


class Mamba2Backbone(SequenceBackbone):
    """Mamba2 主干包装器：将 mamba_ssm.Mamba2 的接口统一为 SequenceBackbone 协议。
    支持变长序列 mask 处理 padding。"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_layers=1,
                 dropout=0.1, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        if _HAS_MAMBA_SSM and _Mamba2Native is not None:
            try:
                self.mixer = _Mamba2Native(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                    layer_idx=layer_idx,
                )
            except TypeError:
                self.mixer = _Mamba2Native(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                )
        else:
            self.mixer = _SS6Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
        self.norm = _RMSNorm(d_model) if _HAS_MAMBA_SSM else nn.LayerNorm(d_model)

    def _run_mixer(self, x, state=None):
        if isinstance(self.mixer, _SS6Block):
            return self.mixer(x, state=state)
        out = self.mixer(x)
        if isinstance(out, tuple):
            return out[0], (out[1] if len(out) > 1 else None)
        return out, None

    def forward(self, x, lengths=None, state=None):
        """x: [B, T, D] -> (out [B, T, D], new_state)
        lengths: [B] 有效长度（用于 mask）
        state: 传递给 mixer 的隐状态（SS6 路径有效；原生 Mamba2 忽略）
        """
        B, T, D = x.size()
        if lengths is not None:
            lengths_dev = lengths.to(device=x.device)
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
            x_masked = x.clone()
            x_masked[~mask] = 0
            out, new_state = self._run_mixer(x_masked, state=state)
            out = out * mask.unsqueeze(-1)
        else:
            out, new_state = self._run_mixer(x, state=state)
        out = self.norm(out + x)
        return out, new_state

    def state_to_dict(self):
        return None

    def state_from_dict(self, state_dict):
        pass


class _SS6Block(nn.Module):
    """S6 简化实现（回退用）：带隐状态递推的 SSM 块。
    基于 S6 (State Space Model) 的简化版本，实现真正的状态递推。
    状态 h in R^{d_inner x d_state} 在每个时间步更新，支持跨 forward 调用持久化。
    SSM 回退实现精度低于 Mamba2 原生，compare 命令会单独标注当前用的是哪种。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, padding=0, groups=self.d_inner)
        # SSM 参数
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.dA = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, state=None):
        """x: [B, T, D] -> (out [B, T, D], new_state)
        state: hidden state h in R^{B x d_inner x d_state} or None (zero init)
        """
        B, T, D = x.size()
        x_orig = x  # residual

        # 前馈 + 门控
        xy = self.in_proj(x)  # [B, T, d_inner*2]
        x, gate = xy.chunk(2, dim=-1)
        # 1D 卷积（因果）
        x = x.transpose(1, 2)
        x = F.pad(x, (self.d_conv - 1, 0))
        x = self.conv1d(x)
        x = F.silu(x)
        x = x.transpose(1, 2)

        # SSM 逐时间步递推
        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        else:
            h = state

        dt = F.softplus(self.dt_proj(x) + self.dt_bias)  # [B, T, d_inner]
        B_param = self.B_proj(x)  # [B, T, d_state]
        C_param = self.C_proj(x)  # [B, T, d_state]

        outputs = []
        for t in range(T):
            dt_t = dt[:, t, :]                                    # [B, d_inner]
            B_t = B_param[:, t, :]                                # [B, d_state]
            C_t = C_param[:, t, :]                                # [B, d_state]
            u_t = x[:, t, :]                                      # [B, d_inner]  ← SSM 输入
            # dA 是 [d_inner, d_state]，逐元素乘 dt 得到 [B, d_inner, d_state]
            dA_dt = (self.dA.unsqueeze(0) * dt_t.unsqueeze(-1))   # [B, d_inner, d_state]
            A_bar = torch.exp(-F.softplus(dA_dt))                 # [B, d_inner, d_state]
            # B_bar = dt * B * u_t，形状 [B, d_inner, d_state]
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
            h = A_bar * h + B_bar
            # 输出 y_t = C^T * h_t，形状 [B, d_inner]
            y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)              # [B, d_inner]
            outputs.append(y_t.unsqueeze(1))

        x_out = torch.cat(outputs, dim=1) * F.silu(gate)
        x_out = self.out_proj(x_out)
        return x_out, h

    def state_to_dict(self):
        return None

    def state_from_dict(self, state_dict):
        pass


class _ExpertLSTMPool(nn.Module):
    """内容通道「纯专家」LSTM 池：序列级 Top-k 路由，真实 batch 切分 dispatch。
    与 FFN-MoE 的区别：LSTM 的算力在隐层循环上，token 级掩码省不了算力，
    所以这里按「整条序列归哪个专家」把 batch 切开分别喂 —— 每个专家只看到
    路由给它的子 batch，激活算力 = Top-k/总专家数 × 稠密等价物，是真省。
    """

    def __init__(self, input_dim, hidden, num_experts=4, top_k=1,
                 dropout=0.1, aux_weight=0.01, min_load=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_weight = aux_weight
        self.min_load = min_load
        self.min_load_weight = getattr(config, "MIN_LOAD_WEIGHT", 0.5)
        self.hidden = hidden
        # 路由器：用输入序列均值打分（不经过 LSTM，路由本身几乎零成本）
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
            for _ in range(num_experts)
        ])
        self.last_usage = None   # [E] 本 batch 平均激活权重（报告用，不带梯度）

    def expert_param_count(self):
        return sum(p.numel() for e in self.experts for p in e.parameters())

    def forward(self, x, lengths, total_length=None):
        """x: [B, T, I]，lengths: [B] -> (out [B, T, H], aux, usage[E])"""
        B, T, H = x.size(0), x.size(1), self.hidden
        probs = F.softmax(self.router(x.mean(dim=1)), dim=-1)      # [B, E]
        topk_w, topk_idx = _topk_route(probs, self.top_k)
        out = x.new_zeros(B, T, H)
        for e_id in range(self.num_experts):
            hit = (topk_idx == e_id)                               # [B, k]
            rows = torch.where(hit.any(dim=1))[0]
            if rows.numel() == 0:
                continue
            # 同一序列在 k 个槽位里命中同一专家时权重相加（正常 top-k 不会重复）
            w_e = (topk_w[rows] * hit[rows]).sum(dim=1, keepdim=True)   # [n, 1]
            o, _ = _packed_lstm(
                self.experts[e_id], x[rows], lengths.detach().cpu()[rows.detach().cpu()],
                total_length=total_length
            )
            out.index_add_(0, rows, w_e.unsqueeze(1) * o)
        aux = _load_balance_aux(probs, self.num_experts, self.aux_weight,
                                self.min_load, self.min_load_weight)
        self.last_usage = probs.detach().mean(dim=0)   # 留在设备上，体检时才 .cpu()
        return out, aux, self.last_usage


class _SemiExpertLSTM(nn.Module):
    """情感通道「半专家」支路：Top-k 小 LSTM 专家，只在完整模型底座上做修正。
    与 _ExpertLSTMPool 的两个关键差异：
      1. 不含主干 —— 主干（self.lstm_b）在模型层始终激活，且独占跨轮 (h, c)；
         半专家是**单轮特化**，每轮零初始化，不写回持久化情感状态。
         这样即使某个半专家没被路由到，跨轮情感也绝不会断流。
      2. 输出投影零初始化 —— 加载旧 checkpoint 后半专家贡献严格为 0，
         模型行为与升级前逐位一致；随训练（先投影权重非零，再回传专家）逐步显式化。
    """

    def __init__(self, input_dim, hidden, out_dim, num_experts=2, top_k=1,
                 aux_weight=0.01, min_load=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_weight = aux_weight
        self.min_load = min_load
        self.min_load_weight = getattr(config, "MIN_LOAD_WEIGHT", 0.5)
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
            for _ in range(num_experts)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(hidden, out_dim) for _ in range(num_experts)
        ])
        with torch.no_grad():
            for p in self.out_projs:
                nn.init.zeros_(p.weight)
                nn.init.zeros_(p.bias)
        self.last_usage = None

    def expert_param_count(self):
        return sum(p.numel() for e in self.experts for p in e.parameters())

    def forward(self, x, lengths, total_length=None):
        """x: [B, T, I] -> (mix [B, T, out_dim], aux, usage[E])"""
        B, T = x.size(0), x.size(1)
        out_dim = self.out_projs[0].out_features
        probs = F.softmax(self.router(x.mean(dim=1)), dim=-1)
        topk_w, topk_idx = _topk_route(probs, self.top_k)
        mix = x.new_zeros(B, T, out_dim)
        for e_id in range(self.num_experts):
            hit = (topk_idx == e_id)
            rows = torch.where(hit.any(dim=1))[0]
            if rows.numel() == 0:
                continue
            w_e = (topk_w[rows] * hit[rows]).sum(dim=1, keepdim=True)
            o, _ = _packed_lstm(
                self.experts[e_id], x[rows], lengths.detach().cpu()[rows.detach().cpu()],
                total_length=total_length
            )
            mix.index_add_(0, rows, w_e.unsqueeze(1) * self.out_projs[e_id](o))
        aux = _load_balance_aux(probs, self.num_experts, self.aux_weight,
                                self.min_load, self.min_load_weight)
        self.last_usage = probs.detach().mean(dim=0)   # 留在设备上，体检时才 .cpu()
        return mix, aux, self.last_usage


class _MoEEncoderLayer(nn.Module):
    """TF-1 检索编码器层（专家化）：多头自注意力 + Top-k 专家 FFN。
    与 _MoEDecoderLayer 同一套专家骨架，只是没有交叉注意力 —— 这样
    「检索」和「生成」两条通道的专家分工能在同一语义下互相比较。
    """

    def __init__(self, d_model, nhead, num_experts=4, top_k=2,
                 expert_ff_dim=1536, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe_ffn = _MoELayer(d_model, num_experts, top_k, expert_ff_dim, dropout,
                                 aux_loss_weight=getattr(config, "ENC_LOAD_BALANCE_WEIGHT", 0.01))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        sa, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(sa))
        ff, aux = self.moe_ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x, aux


class _MoEEncoder(nn.Module):
    """专家化编码器（替掉 nn.TransformerEncoder）：forward 多返回一个 aux 合计。"""

    def __init__(self, d_model, nhead, num_layers, num_experts=4, top_k=2,
                 expert_ff_dim=1536, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _MoEEncoderLayer(d_model, nhead, num_experts, top_k, expert_ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, src_key_padding_mask=None):
        aux_total = x.new_zeros(())
        for layer in self.layers:
            x, aux = layer(x, key_padding_mask=src_key_padding_mask, attn_mask=mask)
            aux_total = aux_total + aux
        return x, aux_total


class ContentMoEBackbone(nn.Module):
    """内容通道：双向专家 Transformer。每 token 看见整段上下文（含多段打包），
    FFN 走 Top-k 专家，注意力是标准 MultiheadAttention（手机端可用）。"""

    def __init__(self, d_model, nhead, num_layers=2, num_experts=4, top_k=2,
                 expert_ff_dim=1536, dropout=0.1, max_segs=4):
        super().__init__()
        self.max_segs = max_segs
        self.seg_embed = nn.Embedding(max_segs, d_model)
        self.pos = PositionalEncoding(d_model, dropout=0.0)
        self.encoder = _MoEEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            num_experts=num_experts, top_k=top_k,
            expert_ff_dim=expert_ff_dim, dropout=dropout,
        )

    def forward(self, x, src=None, pad_mask=None):
        """x: [B, T, D] -> (out [B, T, D], aux)"""
        if src is not None:
            seg_ids = _segment_ids_from_src(src, self.max_segs)
            x = x + self.seg_embed(seg_ids)
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=pad_mask)


class PositionalEncoding(nn.Module):
    """正弦位置编码（无参数，运行时按实际序列长度动态计算，无固定上限）"""

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        position = torch.arange(0, T, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, T, self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term[: pe.size(-1) // 2])
        return self.dropout(x + pe)


class UnifiedAnwanModel(nn.Module):
    """
    一体化统一模型：内容通道(双向专家 Transformer) / LSTM-B(情感·跨轮) / TF-1(检索) / TF-2(生成)
    同一实例四组参数块，共享 embedding + 输出投影；中央 soft router 自动分区。
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=1536,
        lstm_hidden=8192,
        lstm_b_hidden=3600,
        emotion_dim=512,
        num_heads=12,
        trans_layers=5,
        decoder_layers=4,
        ff_dim=6144,
        emotion_vec_dim=3,
        dropout=0.1,
        num_intents=14,
        tie_weights=True,
        cross_gate_init=0.1,
        entropy_floor=0.8,
        router_entropy_weight=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.emotion_dim = emotion_dim
        self.max_len = getattr(config, "MAX_LEN", 80)
        self.num_intents = num_intents
        self.entropy_floor = entropy_floor
        self.router_entropy_weight = router_entropy_weight

        # ---- 共享 embedding（四组模块共用同一张表）----
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # ---- 内容通道（无状态 / 单轮，可一次吃多段文本）----
        # 默认 tf：双向专家 Transformer（标准注意力，手机端可用）。
        # ssm / lstm 仍作回退；MOE_SCOPE="decoder" 时走稠密 LSTM。
        self.lstm_a_hidden = lstm_hidden
        self.moe_scope = getattr(config, "MOE_SCOPE", "decoder")
        self.lstm_a_pool = None
        self.lstm_a = None
        self.ssm_backbone = None
        self.content_tf = None
        backbone_type = getattr(config, "ANWAN_BACKBONE", "tf")
        if backbone_type == "tf" or (backbone_type not in ("ssm", "lstm") and self.moe_scope == "all"):
            self.content_tf = ContentMoEBackbone(
                d_model=embed_dim, nhead=num_heads,
                num_layers=getattr(config, "CONTENT_TF_LAYERS", 2),
                num_experts=getattr(config, "CONTENT_TF_EXPERTS", 4),
                top_k=getattr(config, "CONTENT_TF_TOP_K", 2),
                expert_ff_dim=getattr(config, "CONTENT_TF_FF_DIM", 1536),
                dropout=dropout,
                max_segs=getattr(config, "CONTENT_PACK_MAX_SEGS", 4),
            )
            content_dim = embed_dim
        elif backbone_type == "ssm" and self.moe_scope == "all":
            ssm_d_model = getattr(config, "EMBED_DIM", 1536)
            self.ssm_backbone = Mamba2Backbone(
                d_model=ssm_d_model, d_state=16, d_conv=4, expand=2,
                num_layers=1, dropout=config.DROPOUT, layer_idx=0,
            )
            content_dim = ssm_d_model
        elif self.moe_scope == "all":
            self.lstm_a_pool = _ExpertLSTMPool(
                embed_dim, getattr(config, "LSTM_A_EXPERT_HIDDEN", lstm_hidden),
                num_experts=config.LSTM_A_EXPERTS, top_k=config.LSTM_A_TOP_K,
                aux_weight=config.LSTM_A_LOAD_BALANCE_WEIGHT,
                min_load=getattr(config, "LSTM_A_MIN_LOAD", 0.0),
            )
            content_dim = self.lstm_a_pool.hidden
        else:
            self.lstm_a = nn.LSTM(
                embed_dim, lstm_hidden, num_layers=1, batch_first=True, bidirectional=False
            )
            content_dim = lstm_hidden

        # ---- LSTM-B：情感通道（有状态 / 跨轮 / emotion_dim）----
        # 800M 规格：emb_proj(1536→512) + LSTM(512→3600) + out_proj(3600→512)
        self.emb_proj = nn.Linear(embed_dim, emotion_dim)
        self.lstm_b = nn.LSTM(
            emotion_dim, lstm_b_hidden, num_layers=1, batch_first=True, bidirectional=False
        )
        self.emo_out_proj = nn.Linear(lstm_b_hidden, emotion_dim)
        self.lstm_b_hidden = lstm_b_hidden
        # ---- 情感通道半专家支路（v3：完整模型 + 半专家）----
        # 完整模型 self.lstm_b 保持原名/原形状：旧 checkpoint 直接命中，且它是
        # 唯一持有跨轮 (h, c) 的主干；半专家是单轮特化支路，输出投影零初始化，
        # 因此加载旧模型后情感行为与升级前逐位一致，随训练逐步显式化。
        self.lstm_b_semi = None
        self.emo_semi_gate_max = getattr(config, "EMO_SEMI_GATE_MAX", 0.5)
        if self.moe_scope == "all" and getattr(config, "EMO_FULL_MODEL", True):
            self.lstm_b_semi = _SemiExpertLSTM(
                emotion_dim, getattr(config, "EMO_SEMI_EXPERT_HIDDEN", 1200), lstm_b_hidden,
                num_experts=config.EMO_SEMI_EXPERTS, top_k=config.EMO_SEMI_TOP_K,
                aux_weight=config.EMO_LOAD_BALANCE_WEIGHT,
                min_load=getattr(config, "EMO_SEMI_MIN_LOAD", 0.0),
            )
            self.emo_semi_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "EMO_SEMI_GATE_INIT", 0.15)))
            )
        # ---- 显式情感向量头（PAD 三维：愉悦/唤醒/支配）----
        # 从 LSTM-B 池化输出回归 3 维可解释向量，tanh 输出 ∈ [-1,1]。
        # 关键：heads 权重零初始化 —— 旧 checkpoint 迁移后情感向量从 ~0 起步，
        # 不改变既有 logits 分布，随训练逐步显式化，安全兼容旧模型。
        self.emotion_vec_dim = emotion_vec_dim
        self.emotion_vec_head = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.SiLU(),
            nn.Linear(64, emotion_vec_dim),
        )
        # 上一轮情感向量 → 生成条件注入口（零初始化，训练中逐步激活）
        self.vec_proj = nn.Linear(emotion_vec_dim, embed_dim)
        with torch.no_grad():
            nn.init.zeros_(self.emotion_vec_head[-1].weight)
            nn.init.zeros_(self.emotion_vec_head[-1].bias)
            nn.init.zeros_(self.vec_proj.weight)
            nn.init.zeros_(self.vec_proj.bias)

        # ---- 通道投影（LSTM 输出 → Transformer 维度）----
        self.content_proj = nn.Linear(content_dim, embed_dim)
        self.emotion_proj = nn.Linear(emotion_dim, embed_dim)

        # ---- TF-1：检索编码器 ----
        # ---- TF-1：检索编码器（v3：FFN 换成 Top-k 专家池）----
        if self.moe_scope == "all":
            self.tf1 = _MoEEncoder(
                d_model=embed_dim, nhead=num_heads, num_layers=trans_layers,
                num_experts=config.ENC_EXPERTS, top_k=config.ENC_TOP_K,
                expert_ff_dim=config.ENC_EXPERT_FF_DIM, dropout=dropout,
            )
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True, activation="relu",
            )
            self.tf1 = nn.TransformerEncoder(enc_layer, num_layers=trans_layers)

        # ---- TF-2：生成解码器（MoE 版）----
        # 每 decoder 层保留自注意力+交叉注意力，FFN 替换为 Top-k 专家池（轻量）。
        # 不继承 nn.TransformerDecoderLayer（避免冗余线性层参数），直接手写。
        self.tf2 = nn.ModuleList([
            _MoEDecoderLayer(
                d_model=embed_dim, nhead=num_heads,
                num_experts=config.NUM_EXPERTS, top_k=config.TOP_K_EXPERTS,
                expert_ff_dim=config.EXPERT_FF_DIM, dropout=dropout,
            )
            for _ in range(decoder_layers)
        ])

        # ---- 位置编码（解码器使用；编码器由 LSTM 自带时序）----
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        # ---- 共享输出投影（可选 tie 到 embedding 权重）----
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.output_proj.weight = self.shared_embedding.weight

        # ---- 渐进入口参数 ----
        # 交叉辅通路弱门控（0.1 起步，可学习，上限由 forward clamp 控制）
        self.cross_gate = nn.Parameter(torch.tensor(float(cross_gate_init)))
        # 情感→生成 主通路强度（1.0 起步，可学习）
        self.emotion_gate = nn.Parameter(torch.tensor(1.0))

        # ---- 中央自动分区 router（soft 门控，非 top-k）----
        # 四个模块各出一个分量表示，router 决定 softmax 组合权重（分区由训练自动学出）
        self.module_projs = nn.ModuleList([
            nn.Linear(content_dim, embed_dim),   # 内容通道（tf 时即 embed_dim）
            nn.Linear(emotion_dim, embed_dim),   # LSTM-B 情感
            nn.Linear(embed_dim, embed_dim),     # TF-1 检索
            nn.Linear(embed_dim, embed_dim),     # TF-2 生成
        ])
        self.router_net = nn.Linear(embed_dim * 4, 4)  # softmax 组合权重

        # ---- 一体化意图头（14 意图多任务监督，train 可选）----
        self.intent_head = nn.Linear(embed_dim, num_intents)

        self.dropout = nn.Dropout(dropout)
        # ---- 分化锁定（功能区固化）----
        self._locked_mask = None  # 训练后固化的硬门控掩码 [4]；None=未锁定
        # ---- AWIL 内部通信总线（零新依赖，旁路日志/调试）----
        self.awil_bus = AWILBus(capacity=2048)
        self._init_weights()

    def _init_weights(self):
        # 半专家输出投影 / PAD 情感头 / 向量注入 / SSM 内部参数必须保持零初始化（旧 ckpt 迁移后
        # 行为不变的保证），这里显式豁免，否则 xavier 会把「贡献严格为 0」冲成随机。
        skip = ("emotion_vec_head", "vec_proj", "lstm_b_semi.out_projs",
                "ssm_backbone", "ssm_backbone.mixer")
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "output_proj" not in name and not any(s in name for s in skip):
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------ 推理辅助 ----------
    def _module_vectors(self, content_out, emotion_out, mem, dec_out):
        """router 输入：四模块序列表示的时序均值 + 各自投影到 embed_dim"""
        feats = []
        for proj, feat in zip(
            self.module_projs,
            [content_out.mean(dim=1), emotion_out.mean(dim=1), mem.mean(dim=1), dec_out.mean(dim=1)],
        ):
            feats.append(proj(feat))
        return feats  # list of [B, embed_dim]

    def route(self, feats):
        """竞争门控（自组织分化版）：
        - 训练：soft 加权 + 正则约束（竞争 + 均值 + 正交）
        - 推理且已锁定：argmax 硬门控，只激活主导模块
        - 返回 weights + entropy + mean_reg + ortho_reg
        """
        router_in = torch.cat(feats, dim=-1)  # [B, 4*E]
        logits = self.router_net(self.dropout(router_in))
        # 竞争强度随 epoch 自适应（外部通过 _competition_temp 注入）
        temp = getattr(self, "_competition_temp", 1.0)
        weights = F.softmax(logits / temp, dim=-1)  # [B, 4]
        avg_w = weights.mean(dim=0)          # [4]
        entropy = -(avg_w * torch.log(avg_w.clamp_min(1e-9))).sum()
        # 自组织均值约束：模块0/1一组，模块2/3一组，组内互斥
        group0 = avg_w[0:2].sum()  # 内容+情感
        group1 = avg_w[2:4].sum()  # 检索+生成
        mean_reg = ((group0 - group1).abs() - 0.2).clamp_min(0.0)
        # 模块正交约束：各模块输出表示的余弦相似度惩罚
        if len(feats) >= 2:
            stacked = torch.stack(feats, dim=0)  # [4, B, E]
            normed = F.normalize(stacked, dim=-1)
            sim_mat = torch.mm(normed.mean(dim=1), normed.mean(dim=1).t())  # [4,4]
            ortho_reg = (sim_mat ** 2).sum() - 4.0  # 对角线=1，非对角线趋近0
            ortho_reg = ortho_reg.clamp_min(0.0)
        else:
            ortho_reg = torch.tensor(0.0, device=logits.device)
        # 硬门控：推理模式且已锁定时，按固化掩码限制主导模块
        if not self.training and self._locked_mask is not None:
            mask = self._locked_mask.to(device=weights.device, dtype=weights.dtype).unsqueeze(0)  # [1,4]
            masked = weights * mask
            s = masked.sum(dim=-1, keepdim=True)
            # 掩码全零时退回原权重，避免除零
            weights = torch.where(s > 1e-9, masked / s.clamp_min(1e-9), weights)
            hard = torch.zeros_like(weights)
            hard.scatter_(1, weights.argmax(dim=1, keepdim=True), 1.0)
            weights = hard
        return weights, entropy, mean_reg, ortho_reg

    @torch.no_grad()
    def lock_specialization(self, tokenizer, texts_by_intent, device="cpu"):
        """分化锁定：根据语料评估各意图类型的主导模块，固化硬门控掩码。
        texts_by_intent: dict[str, list[str]]，如 {"greeting": ["你好", "嗨"], ...}
        返回特化报告 dict{"intents": {"greeting": {"top_module": 2, "top_weight": 0.65, ...}},
                           "global": {"entropy": 0.8, "locked": True}}
        """
        self.eval()
        report = {"intents": {}, "global": {"entropy": None, "locked": False}}
        if not texts_by_intent:
            return report
        all_ws = []
        for intent, texts in texts_by_intent.items():
            ws = []
            for t in texts:
                ids = tokenizer.encode(t)
                src = torch.tensor([ids], device=device)
                tgt = torch.tensor([[BOS_IDX]], device=device)
                out = self(src, tgt, emotion_state=None, stage="main", return_intent=False)
                ws.append(out["router_weights"][0].cpu())
            avg = torch.stack(ws).mean(dim=0)
            top = int(avg.argmax())
            report["intents"][intent] = {
                "top_module": top,
                "top_weight": float(avg[top]),
                "weights": [float(v) for v in avg],
            }
            all_ws.extend(ws)
        if all_ws:
            global_avg = torch.stack(all_ws).mean(dim=0)
            report["global"]["entropy"] = float(-(global_avg * torch.log(global_avg.clamp_min(1e-9))).sum())
            report["global"]["avg_load"] = [float(v) for v in global_avg]
            # 锁定：计算全局主导模块掩码（取全局平均 top-2 置 1）
            mask = torch.zeros(4, device=device)
            top2 = global_avg.topk(2).indices
            mask[top2] = 1.0
            # v3：情感模块（下标 1）强制常驻 —— 它是唯一持有跨轮 (h,c) 与 PAD 向量
            # 的完整模型，被中央 router 抑制等于跨轮人设直接断电，比路由坍缩更致命。
            mask[1] = 1.0
            self._locked_mask = mask
            report["global"]["locked"] = True
            report["global"]["mask"] = [int(v) for v in mask.cpu()]
        return report

    # ------------------------------------------------ 主干 --------------
    def forward(
        self,
        src,
        tgt_in,
        emotion_state=None,
        stage="main",
        return_intent=False,
        intent_labels=None,
        prev_emotion_vec=None,
    ):
        """
        一次前向（teacher forcing）：
        - src:     [B, T] 问题 token（含 BOS）
        - tgt_in:  [B, T2] 回复 token 前缀（含 BOS；训时右移一位）
        - emotion_state: LSTM-B 的 (h, c)，跨轮传递；None 则零初始化
        - prev_emotion_vec: 上一轮显式情感向量 [B,3]/[1,3]/None，注入生成条件
        - stage:   "main"（对角主通路）| "cross"（+交叉辅通路弱门控）
        - return_intent: 是否输出 intent logits（train 监督用）
        返回 dict: logits / new_state / router_weights / router_entropy / intent_logits
        """
        device = src.device
        B, T = src.size()
        T2 = tgt_in.size(1)
        lengths = _effective_lengths(src)             # [B] 有效 token 数

        src_emb = self.shared_embedding(src)          # [B, T, E]

        # ---- 内容通道（双向专家 Transformer / SSM / LSTM；可一次吃多段）----
        content_out, aux_content, usage_content = self._encode_content(
            src_emb, lengths, T, src=src
        )                                             # [B, T, content_dim]

        # ---- LSTM-B：情感通道（有状态，跨轮；仅有效 token 演化）----
        if emotion_state is not None:
            h0, c0 = emotion_state
        else:
            h0 = torch.zeros(1, B, self.lstm_b_hidden, device=device)
            c0 = torch.zeros(1, B, self.lstm_b_hidden, device=device)
        emotion_out, (hB, cB), aux_emo, usage_emo = self._encode_emotion(
            self.emb_proj(src_emb), lengths, T, h0, c0
        )                                             # [B, T, emotion_dim]
        new_state = (hB.detach().clone(), cB.detach().clone())

        # ---- 对角主通路：内容 → 检索（TF-1）----
        mem_in = self.content_proj(content_out)       # [B, T, E]

        # ---- 交叉辅通路：情感 → 检索（stage=cross 且弱门控起步）----
        if stage == "cross":
            emo_pool = emotion_out.mean(dim=1)        # [B, emotion_dim]
            gate = self.cross_gate.clamp(max=0.5)
            mem_in = mem_in + gate * self.emotion_proj(emo_pool).unsqueeze(1)

        src_mask = _pad_mask(src, device)
        mem, aux_enc = self._encode_mem(mem_in, src_mask)       # [B, T, E]

        # ---- 显式情感向量（P4）：PAD 语义头 + 跨轮注入 ----
        emo_pool = emotion_out.mean(dim=1)                     # [B, emotion_dim]
        emotion_vec = torch.tanh(self.emotion_vec_head(emo_pool))  # [B, 3] ∈ [-1,1]
        # ---- 对角主通路：情感 → 生成（把情感摘要注入检索 memory）----
        emo_cond = self.emotion_proj(emo_pool)                 # [B, E]
        if prev_emotion_vec is not None:
            pv = prev_emotion_vec
            if pv.dim() == 1:
                pv = pv.unsqueeze(0).expand(B, -1)
            if pv.size(0) != B:
                pv = pv[:B] if pv.size(0) > B else pv[0:1].expand(B, -1)
            # 上一轮 PAD 向量作为跨轮情感语境注入（训练时随 batch 回喂）
            vec_cond = self.vec_proj(pv)
            emo_cond = emo_cond + config.EMOTION_VEC_INJECT * vec_cond
        mem = mem + self.emotion_gate * emo_cond.unsqueeze(1) * 0.4

        # ---- TF-2：生成（MoE 解码器层，逐层传递并收集 aux_loss）----
        tgt_emb = self.pos_encoding(self.shared_embedding(tgt_in))  # [B, T2, E]
        if stage == "cross":
            content_pool = content_out.mean(dim=1)    # [B, lstm_hidden]
            gate = self.cross_gate.clamp(max=0.5)
            tgt_emb = tgt_emb + gate * self.content_proj(content_pool).unsqueeze(1) * 0.4

        tgt_mask = _build_causal_mask(T2, device)
        tgt_pad_mask = _pad_mask(tgt_in, device)
        dec_out = tgt_emb
        moe_aux_losses = []
        for dec_layer in self.tf2:
            dec_out, aux = dec_layer(
                dec_out, mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_mask,
            )
            moe_aux_losses.append(aux)
        # MoE 负载均衡辅助 loss 总和
        for _aux in (aux_content, aux_emo, aux_enc):
            if _aux is not None:
                moe_aux_losses.append(_aux)
        moe_aux_total = sum(moe_aux_losses) if moe_aux_losses else torch.tensor(0.0, device=device)

        # ---- 中央 router：竞争门控（自组织分化）----
        feats = self._module_vectors(content_out, emotion_out, mem, dec_out)
        weights, entropy, mean_reg, ortho_reg = self.route(feats)  # [B,4] / scalar / reg / reg

        fused = torch.stack(feats, dim=1)            # [B, 4, E]
        fused = (fused * weights.unsqueeze(-1)).sum(dim=1)  # [B, E] soft 加权融合

        # ---- 共享输出投影 ----
        logits = self.output_proj(dec_out + fused.unsqueeze(1))  # [B, T2, vocab]

        result = {
            "logits": logits,
            "new_state": new_state,
            # 情绪方向 loss 需要**带梯度**的进出状态；new_state 已 detach，
            # 梯度到不了 LSTM-B，这里单独暴露（只给损失用，不外传）
            "emo_h_in": h0 if emotion_state is not None else torch.zeros_like(h0),
            "emo_h_out": hB,
            "emotion_vec": emotion_vec,
            "moe_aux_loss": moe_aux_total,
            "expert_usage": (self.expert_usage()
                             if getattr(self, "_collect_expert_usage", False) else {}),
            "router_weights": weights,
            "router_entropy": entropy,
            "router_mean_reg": mean_reg,
            "router_ortho_reg": ortho_reg,
        }

        # ---- 意图头（可选监督）----
        if return_intent or intent_labels is not None:
            intent_logits = self.intent_head(fused)   # [B, num_intents]
            result["intent_logits"] = intent_logits

        # ---- AWIL：发射路由 + 意图消息（旁路日志）----
        # 训练期默认关闭：weights[0].cpu() 在 GPU 上是强制同步，每步一次会拖慢训练，
        # 与"零性能影响"的说法不符。推理期（eval 模式）保留完整流水账。
        if (getattr(self, "awil_bus", None) is not None and self.awil_bus.enabled
                and not (self.training and not config.AWIL_ENABLED_IN_TRAIN)):
            self.awil_bus.emit_route(
                AWIL.MOD_ROUTER, AWIL.MOD_BUS,
                weights[0].cpu().tolist(), float(entropy.cpu().item())
            )
            if "intent_logits" in result:
                intent_id = int(result["intent_logits"][0].argmax().cpu().item())
                intent_conf = float(F.softmax(result["intent_logits"][0], dim=0)[intent_id].cpu().item())
                self.awil_bus.emit_intent(
                    AWIL.MOD_INTENT, AWIL.MOD_BUS, intent_id, intent_conf
                )

        return result

    # ------------------------------------------------ 情绪方向 ----------
    @staticmethod
    def emotion_arousal(intent_name):
        """各意图的"情绪唤醒度"目标：0=平稳，1=剧烈波动。
        规则版，与 UnifiedIntentLabeler 的意图语义对齐。"""
        return config.EMOTION_AROUSAL.get(intent_name, 0.35)

    def emotion_direction_loss(self, out, intent_labels):
        """
        让情感状态的**变化幅度**与意图的唤醒度匹配：
          调侃/倾诉/彩蛋 → 状态应该明显变化
          闲聊/提问/查询 → 状态应该基本平稳
        用相对变化率 ratio = ‖h_out − h_in‖ / (‖h_in‖ + ε) 而非绝对幅度 ——
        LSTM 状态的范数会随对话长度自然增长，用绝对目标会跟这个增长打架，
        训练会不稳定。hinge 形式：只在越过允许区间时才惩罚。
        
        SSM 状态下语义差异说明：
        - LSTM 状态 (h,c) 是显式递归状态，跨步有明确的隐状态传递；
        - SSM (Mamba2) 状态隐含在权重递推中，我们通过取序列末位置输出近似"状态"。
        因此 SSM 路径下的 ‖Δh‖ 度量的是"序列末态与初态的差异"，
        与 LSTM 的"隐状态变化"语义不完全等价，需通过 calibrate-emotion 重标定超参。
        若相对分离度恒 < 0，ANWAN_EMO_DIR_NORM=std 分支会对变化幅度做 z-score 标准化。
        """
        if intent_labels is None or out.get("emo_h_out") is None:
            return None
        w = getattr(config, "EMOTION_DIRECTION_WEIGHT", 0.0)
        if not w:
            return None
        names = config.INTENTS
        h_in = out["emo_h_in"]
        h_out = out["emo_h_out"]
        # 每样本：状态变化比例。状态是 [layers=1, B, H]，先在层维聚合成 [B]，
        # 否则后面按 batch 下标索引会越界。
        delta = (h_out - h_in).norm(dim=-1).mean(dim=0)    # [B]
        hn_in = h_in.norm(dim=-1).mean(dim=0)              # [B]
        hn_out = h_out.norm(dim=-1).mean(dim=0)            # [B]
        # 分母必须留地板：首轮 h_in 全零，若用 ‖h_in‖+ε 做分母，比值会飙到
        # 上千（实测 ~3000），直接把训练打崩。改用 ‖h_out‖ 兜底并 clamp 到 1，
        # 零状态时比值退化为 ‖Δh‖（有界、稳定），非零时仍是相对量。
        scale = torch.clamp(hn_in + hn_out, min=1.0)
        ratio = delta / scale
        # ANWAN_EMO_DIR_NORM=std 时先对状态变化幅度做标准化再比较
        # （解决 SSM 状态下相对分离度恒 < 0 的问题）
        emo_dir_norm = getattr(config, "ANWAN_EMO_DIR_NORM", "std")
        if emo_dir_norm == "std" and len(delta) > 1:
            # 对 ratio 做 z-score 标准化（按 batch 维度）
            ratio_mean = ratio.mean()
            ratio_std = ratio.std().clamp_min(1e-6)
            ratio = (ratio - ratio_mean) / ratio_std
        active = torch.zeros_like(ratio, dtype=torch.bool)
        for i, idx in enumerate(intent_labels.tolist()):
            if 0 <= idx < len(names) and self.emotion_arousal(names[idx]) >= 0.6:
                active[i] = True
        if not active.any() or active.all():
            return None                      # batch 里只有一种唤醒度，无从对比
        # **直接优化两组的相对分离度**，不设绝对阈值。
        # 变化率的天然尺度随模型动力学变化（随机初始化实测 ~0.33），绝对阈值要么
        # 永不触发（loss 恒 0、塑形失效），要么暴涨把训练打崩；逐样本 hinge 的值
        # 又只有 0.01 量级，乘个小权重后占总 loss 不到万分之一，优化器根本不理它。
        # 这里用 (高唤醒均值 − 低唤醒均值) / (两者之和) 这个 O(1) 的无量纲相对量，
        # 尺度自校准、量级可控，权重才有意义。
        mu_a = ratio[active].mean()
        mu_p = ratio[~active].mean()
        rel_sep = (mu_a - mu_p) / (mu_a + mu_p + 1e-6)      # 约 [-1, 1]
        loss = F.relu(config.EMOTION_AROUSAL_MARGIN - rel_sep)
        return loss * w

    def emotion_vec_loss(self, out, intent_labels, prev_emotion_vec=None):
        """
        显式情感向量监督（P4）——让情感真正有向量且参与训练：
        1) 意图匹配：emotion_vec 回归该意图的 PAD 目标向量（愉悦/唤醒/支配）。
           与 emotion_direction_loss"只约束变化幅度"互补：这里约束的是**方向与
           内容**——(0.5,0.8,0.4) 就是"兴奋"、(-0.2,0.1,0.4) 就是"严肃命令"。
        2) 跨轮连续性：低唤醒意图（闲聊/提问/查询）要求前后轮向量接近，
           高唤醒意图（调侃/倾诉/彩蛋）放行跳变——与唤醒度语义一致。
        返回 (loss, mse, cont) 或 None（无监督信号时）。
        """
        w = getattr(config, "EMOTION_VEC_WEIGHT", 0.0)
        if not w or "emotion_vec" not in out or intent_labels is None:
            return None
        names = getattr(config, "INTENTS", [])
        targets = torch.tensor(
            [config.EMOTION_PAD_TARGETS.get(names[int(i)], (0.0, 0.2, 0.0))
             for i in intent_labels.cpu().tolist()],
            device=out["emotion_vec"].device,
            dtype=out["emotion_vec"].dtype,
        )
        mse = F.mse_loss(out["emotion_vec"], targets)
        total = mse * w
        cont = None
        cw = getattr(config, "EMOTION_VEC_CONT_WEIGHT", 0.0)
        if cw and prev_emotion_vec is not None:
            idxs = intent_labels.cpu().tolist()
            low = [i for i, idx in enumerate(idxs)
                   if 0 <= idx < len(names)
                   and self.emotion_arousal(names[idx]) < config.EMOTION_VEC_CONT_AROUSAL_MAX]
            if low:
                cont = F.mse_loss(out["emotion_vec"][low],
                                  prev_emotion_vec[low].detach())
                total = total + cont * cw
        return total, mse, cont

    # ------------------------------------------------ 损失计算 ----------
    def compute_loss(self, out, tgt_out, intent_labels=None, intent_loss_weight=8.0,
                     stage_weight=1.0, prev_emotion_vec=None):
        """
        组合损失（自组织分化版）：
        - seq_loss：next-token 交叉熵
        - intent_loss：意图头多任务监督
        - router_reg：竞争正则（batch 熵 + 均值约束 + 模块正交）
        """
        logits = out["logits"]
        seq_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=PAD_IDX,
        )
        total = seq_loss

        intent_loss = torch.tensor(0.0, device=logits.device)
        if intent_labels is not None and "intent_logits" in out:
            intent_loss = F.cross_entropy(out["intent_logits"], intent_labels)
            total = total + intent_loss * intent_loss_weight

        entropy = out["router_entropy"]
        reg = (self.entropy_floor - entropy).clamp_min(0.0) * self.router_entropy_weight
        mean_reg = out.get("router_mean_reg", torch.tensor(0.0, device=logits.device))
        ortho_reg = out.get("router_ortho_reg", torch.tensor(0.0, device=logits.device))
        comp_weight = getattr(self, "_competition_weight", 0.1)
        total = total + reg + comp_weight * (mean_reg + ortho_reg)

        # 情绪方向：让情感状态的变化幅度匹配意图唤醒度（塑形项，权重小）
        emo_dir = self.emotion_direction_loss(out, intent_labels)
        if emo_dir is not None:
            total = total + emo_dir
        else:
            emo_dir = torch.tensor(0.0, device=logits.device)

        # 显式情感向量：PAD 三维回归 + 低唤醒意图的跨轮连续性
        emo_vec = torch.tensor(0.0, device=logits.device)
        emo_vec_cont = torch.tensor(0.0, device=logits.device)
        ev_loss = self.emotion_vec_loss(out, intent_labels, prev_emotion_vec)
        if ev_loss is not None:
            ev_total, ev_mse, ev_cont = ev_loss
            total = total + ev_total
            emo_vec = ev_mse.detach()
            if ev_cont is not None:
                emo_vec_cont = ev_cont.detach()

        # MoE 负载均衡辅助 loss（decoder 层专家使用均匀度）
        moe_aux = out.get("moe_aux_loss", torch.tensor(0.0, device=logits.device))
        total = total + moe_aux

        return {
            "total": total,
            "seq_loss": seq_loss.detach(),
            "intent_loss": intent_loss.detach(),
            "emotion_dir": emo_dir.detach(),
            "emotion_vec": emo_vec,
            "emotion_vec_cont": emo_vec_cont,
            "moe_aux": moe_aux.detach(),
            "router_reg": reg.detach(),
            "router_entropy": entropy.detach(),
            "router_mean_reg": mean_reg.detach(),
            "router_ortho_reg": ortho_reg.detach(),
        }


    # ------------------------------------------------ 自回归生成 ----------
    @torch.no_grad()
    def generate(
        self,
        src,
        emotion_state=None,
        max_len=25,
        temperature=0.8,
        top_p=0.9,
        strategy="nucleus",
        stage="main",
        eos_id=EOS_IDX,
        bos_id=BOS_IDX,
        prev_emotion_vec=None,
    ):
        """
        自回归生成：
        - 编码一次（内容通道 + LSTM-B + TF-1），得到 mem 与跨轮情感 state（返回给推理引擎）
        - 解码循环：每个时间步对已生成前缀做一次 TF-2，取末位 logits 采样
        - strategy: "greedy" | "top_k" | "nucleus"
        返回 (token_ids, new_state, router_weights, router_entropy, emotion_vec)
        - router_weights/router_entropy 取解码第 1 步（tgt=[BOS] 等效原 _router_probe，
          受训时一步即得，免除额外二次前向）
        """
        device = src.device
        B, T = src.size()
        lengths = _effective_lengths(src)

        src_emb = self.shared_embedding(src)
        content_out, _, _ = self._encode_content(src_emb, lengths, T, src=src)
        if emotion_state is not None:
            h0, c0 = emotion_state
        else:
            h0 = torch.zeros(1, B, self.lstm_b_hidden, device=device)
            c0 = torch.zeros(1, B, self.lstm_b_hidden, device=device)
        emotion_out, (hB, cB), _, _ = self._encode_emotion(self.emb_proj(src_emb), lengths, T, h0, c0)
        new_state = (hB.clone(), cB.clone())

        mem_in = self.content_proj(content_out)
        if stage == "cross":
            emo_pool = emotion_out.mean(dim=1)
            gate = self.cross_gate.clamp(max=0.5)
            mem_in = mem_in + gate * self.emotion_proj(emo_pool).unsqueeze(1)
        src_mask = _pad_mask(src, device)
        mem, _ = self._encode_mem(mem_in, src_mask)
        emotion_vec = torch.tanh(self.emotion_vec_head(emotion_out.mean(dim=1)))  # [B,3] ∈ [-1,1]
        emo_cond = self.emotion_proj(emotion_out.mean(dim=1))
        if prev_emotion_vec is not None:
            pv = prev_emotion_vec
            if pv.dim() == 1:
                pv = pv.unsqueeze(0).expand(B, -1)
            if pv.size(0) != B:
                pv = pv[:B] if pv.size(0) > B else pv[0:1].expand(B, -1)
            emo_cond = emo_cond + config.EMOTION_VEC_INJECT * self.vec_proj(pv)
        mem = mem + self.emotion_gate * emo_cond.unsqueeze(1) * 0.4

        # router 需要 content/emotion 表示（SEQ 维平均）
        content_pool = content_out.mean(dim=1)
        emotion_pool = emotion_out.mean(dim=1)

        generated = torch.full((B, 1), bos_id, device=device, dtype=torch.long)
        router_weights = None
        router_entropy = None

        for _ in range(max_len):
            T2 = generated.size(1)
            tgt_emb = self.pos_encoding(self.shared_embedding(generated))
            if stage == "cross":
                gate = self.cross_gate.clamp(max=0.5)
                tgt_emb = tgt_emb + gate * self.content_proj(content_pool).unsqueeze(1) * 0.4
            tgt_mask = _build_causal_mask(T2, device)
            tgt_pad_mask = _pad_mask(generated, device)
            # MoE decoder 层逐层前向（推理时 aux_loss 不返回）
            dec_out = tgt_emb
            for dec_layer in self.tf2:
                dec_out, _ = dec_layer(
                    dec_out, mem,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_mask,
                )
            feats = self._module_vectors(content_out, emotion_out, mem, dec_out)
            weights, entropy, _, _ = self.route(feats)
            if router_weights is None:      # 第 1 步即等效原 _router_probe
                router_weights = weights[0].cpu().tolist()
                router_entropy = float(entropy.cpu().item())
            fused = torch.stack(feats, dim=1)         # [B, 4, E]
            fused = (fused * weights.unsqueeze(-1)).sum(dim=1)  # [B, E]
            logits = self.output_proj(dec_out[:, -1:, :] + fused.unsqueeze(1))[:, -1, :]
            logits = logits / max(temperature, 1e-6)

            if strategy == "greedy":
                next_id = logits.argmax(dim=-1, keepdim=True)
            elif strategy == "top_k":
                k = 10
                topk = logits.topk(k, dim=-1)
                probs = F.softmax(topk.values, dim=-1)
                next_id = topk.indices.gather(-1, torch.multinomial(probs, 1))
            else:  # nucleus
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = float("-inf")
                probs = F.softmax(sorted_logits, dim=-1)
                pick = torch.multinomial(probs, 1)
                next_id = sorted_idx.gather(-1, pick)
            generated = torch.cat([generated, next_id], dim=1)
            # ---- AWIL：发射生成步消息（旁路日志）----
            if getattr(self, "awil_bus", None) is not None and self.awil_bus.enabled:
                token_prob = float(F.softmax(logits[0], dim=0)[int(next_id[0, 0].item())].cpu().item())
                self.awil_bus.emit_generate(
                    AWIL.MOD_TF2, AWIL.MOD_BUS, int(next_id[0, 0].item()), token_prob
                )
            if int(next_id[0, 0].item()) == eos_id:
                break

        # ---- AWIL：发射情感状态同步消息 ----
        if getattr(self, "awil_bus", None) is not None and self.awil_bus.enabled:
            emotion_summary = emotion_out.mean(dim=1)[0].cpu().tolist()[:8]  # 只取前8维摘要
            self.awil_bus.emit_emotion(AWIL.MOD_LSTM_B, AWIL.MOD_BUS, emotion_summary)

        return generated, new_state, router_weights, router_entropy, emotion_vec

    # ------------------------------------------------ v3 通道编码 helper ----
    def _encode_content(self, src_emb, lengths, T, src=None):
        """内容通道：双向专家 Transformer / SSM / LSTM 专家池 / 稠密回退
        -> (out, aux|None, usage|None)"""
        if getattr(self, "content_tf", None) is not None:
            pad_mask = _pad_mask(src, src_emb.device) if src is not None else None
            out, aux = self.content_tf(src_emb, src=src, pad_mask=pad_mask)
            return out, aux, None
        backbone_type = getattr(config, "ANWAN_BACKBONE", "tf")
        if backbone_type == "ssm" and self.ssm_backbone is not None:
            out, _ = self.ssm_backbone(src_emb, lengths=lengths)
            return out, None, None
        if getattr(self, "lstm_a_pool", None) is not None:
            return self.lstm_a_pool(src_emb, lengths, total_length=T)
        out, _ = _packed_lstm(self.lstm_a, src_emb, lengths, total_length=T)
        return out, None, None

    def _encode_emotion(self, emo_in, lengths, T, h0, c0):
        """情感通道：完整 LSTM-B（始终激活，独占跨轮 (h,c)）+ 半专家限幅修正。
        返回 (emotion_out [B,T,emotion_dim], (hB,cB), aux|None, usage|None)
        """
        full_out, (hB, cB) = _packed_lstm(
            self.lstm_b, emo_in, lengths, hx=(h0, c0), total_length=T
        )
        aux = usage = None
        if getattr(self, "lstm_b_semi", None) is not None:
            mix, aux, usage = self.lstm_b_semi(emo_in, lengths, total_length=T)
            gate = self.emo_semi_gate.clamp(max=self.emo_semi_gate_max)
            full_out = full_out + gate * mix
        return self.emo_out_proj(full_out), (hB, cB), aux, usage

    def _encode_mem(self, mem_in, src_mask):
        """检索编码器：专家化返回 (mem, aux)，稠密回退 aux=None。"""
        if isinstance(self.tf1, _MoEEncoder):
            return self.tf1(mem_in, src_key_padding_mask=src_mask)
        return self.tf1(mem_in, src_key_padding_mask=src_mask), None

    def expert_usage(self):
        """各专家池的激活分布（体检/防坍缩监控用，全部 detach）。"""
        info = {}
        backbone_type = getattr(config, "ANWAN_BACKBONE", "tf")
        if getattr(self, "content_tf", None) is not None:
            info["backbone"] = "tf(bidirectional-MoE)"
        elif backbone_type == "ssm":
            info["backbone"] = "ssm(Mamba2)" if _HAS_MAMBA_SSM else "ssm(S6-fallback)"
        else:
            info["backbone"] = "lstm"
        pool = getattr(self, "lstm_a_pool", None)
        if pool is not None and pool.last_usage is not None:
            info["lstm_a"] = [round(v, 4) for v in pool.last_usage.detach().cpu().tolist()]
        semi = getattr(self, "lstm_b_semi", None)
        if semi is not None and semi.last_usage is not None:
            info["lstm_b_semi"] = [round(v, 4) for v in semi.last_usage.detach().cpu().tolist()]
            info["lstm_b_gate"] = round(
                float(self.emo_semi_gate.detach().clamp(max=self.emo_semi_gate_max)), 4)
        ctf = getattr(self, "content_tf", None)
        if ctf is not None:
            info["content_tf"] = [
                None if l.moe_ffn.last_usage is None
                else [round(v, 4) for v in l.moe_ffn.last_usage.detach().cpu().tolist()]
                for l in ctf.encoder.layers
            ]
        if isinstance(self.tf1, _MoEEncoder):
            info["tf1"] = [
                None if l.moe_ffn.last_usage is None
                else [round(v, 4) for v in l.moe_ffn.last_usage.detach().cpu().tolist()]
                for l in self.tf1.layers
            ]
        info["tf2"] = [
            None if getattr(l.moe_ffn, "last_usage", None) is None
            else [round(v, 4) for v in l.moe_ffn.last_usage.detach().cpu().tolist()]
            for l in self.tf2
        ]
        return info

    def count_active_parameters(self):
        """激活参数量：专家池里没被路由到的专家不计费（Top-k 稀疏的真实收益）。"""
        total = self.count_parameters(trainable_only=False)
        saved = 0
        for m in self.modules():
            if isinstance(m, (_ExpertLSTMPool, _SemiExpertLSTM, _MoELayer)):
                per_expert = m.expert_param_count() / max(m.num_experts, 1)
                saved += per_expert * max(m.num_experts - m.top_k, 0)
        # SSM 主干参数不计入"节省"（全激活）
        return total - saved

    def count_parameters(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


class UnifiedIntentLabeler:
    """
    一体化意图标注（规则版）：从问题文本标出 14 意图之一
    与旧版 IntentAutoLabeler 规则对齐；供训练流程自动打标签（无需外部意图网络）
    """

    INTENTS = [
        "greeting", "farewell", "identity", "state_query",
        "emotion_share", "affection", "teasing", "question",
        "admin_cmd", "egg_trigger", "action_request", "chat",
        "lang_learning", "rare_word",
    ]
    INTENT2IDX = {name: i for i, name in enumerate(INTENTS)}

    EXACT_RULES = {
        "异环": "egg_trigger", "楚门": "egg_trigger", "斐济": "egg_trigger",
        "42": "egg_trigger", "击毙你": "teasing", "关掉": "teasing",
        "面板": "admin_cmd", "清空记忆": "admin_cmd", "重置感情": "admin_cmd",
        "看日记": "admin_cmd", "自学习": "admin_cmd", "碎语集": "admin_cmd",
        "惩罚记录": "admin_cmd", "缓存统计": "admin_cmd", "词库": "admin_cmd", "随机": "admin_cmd",
    }
    KEYWORD_RULES = {
        "greeting": {"你好", "嗨", "在吗", "早安", "晚安", "早上好", "晚上好", "哈喽", "在不在", "有人吗"},
        "farewell": {"再见", "拜拜", "走了", "去忙", "离开", "回见", "下次见"},
        "identity": {"你叫什么", "你是谁", "名字", "你多大了", "你是什么"},
        "state_query": {"你好吗", "你开心吗", "你在干嘛", "你累吗", "你困不困", "你吃饭了吗", "你醒了吗", "安晚可爱吗", "你有梦想吗", "安晚喜欢什么", "安晚讨厌什么", "安晚开心吗", "安晚在想什么"},
        "emotion_share": {"我累了", "我心情不好", "我想你了", "我不开心", "我难过", "我生气了", "我好开心", "我好幸福", "我好烦", "我想哭", "我被骂了", "我失败了", "我考好了", "我成功了", "我赢了", "我饿了", "我困了", "我睡不着", "失眠了", "我好冷", "我好紧张", "我怕", "我担心", "我好兴奋", "我好孤单", "一个人好无聊", "好无聊啊", "太无聊了", "好孤单"},
        "affection": {"喜欢你", "爱你", "抱抱", "摸摸", "亲亲", "蹭蹭", "一起睡", "永远在一起", "你是我的", "你好可爱", "你好漂亮", "你真贴心", "你最好了", "抱抱我", "抱紧点", "我想抱你", "摸摸我", "我爱你", "你是我最好的朋友", "你是我唯一的朋友", "我喜欢抱着你", "你是我的最爱", "你是我的小宝贝", "你是我的一切", "没有你我怎么办"},
        "teasing": {"你好笨", "你滚吧", "你走吧", "我不理你了", "哼", "你烦不烦", "闭嘴", "我不想听你说话", "你话真多", "你能不能安静点", "你不累吗", "你真娇气", "你太依赖我了", "你太粘人了", "你太吵了", "你能安静一会儿吗", "你真是个小笨蛋", "你这个人真奇怪", "你思维方式好奇怪", "你有时候真让人头疼", "你一点用都没有", "删掉你", "我要删了你", "以后会把你关掉"},
        "question": {"什么", "怎么", "为什么", "多少", "几岁", "几点", "哪里", "是谁", "叫什么", "时间", "天气", "颜色", "英文", "古诗", "数学", "首都", "地球", "太阳", "月亮", "外星人", "鬼", "僵尸", "魔法", "世界末日", "荒岛", "超能力", "怕", "相信"},
        "admin_cmd": {"调快乐值", "调精力值", "调亲密度", "调安全感", "调害羞度", "调依赖度", "写日记", "教我说话", "训练数据", "预览", "重置学习", "导出学习", "训练报告", "粘鼠板", "性格列表", "换性格"},
        "egg_trigger": {"侠岚", "伸出你的左手", "FAIRY", "魔卡少女樱", "哆啦A梦", "蜡笔小新", "你的名字", "男子高中生的日常", "无头骑士异闻录", "命运石之门", "EVA", "坂本", "干物妹小埋", "[up]"},
        "lang_learning": {"怎么翻译", "翻译成", "外语怎么说", "英文怎么说", "这个用英语", "怎么说英文"},
        "rare_word": {"生僻字", "僻字", "生词", "这个词的意思", "字的意思", "查字典"},
        "action_request": {"我想让你开心", "哄哄我", "哄哄你", "陪我聊天", "陪我玩", "给我讲个笑话", "讲个笑话", "背古诗", "背乘法口诀", "变个魔术", "写代码", "生成代码", "你会唱歌吗", "你会背"},
    }
    # 高优先级规则：必须在 KEYWORD_RULES 之前求值。
    # 原因："英文" 原本挂在 question 关键词表里，语言学习类问法
    # （"用英语怎么说"）会先被 question 抢走，标错意图。
    # 这两类问法模式极强（点名语言 + 翻译动词 / 生僻字 + 释义动词），
    # 放在最前面判，冲突就不存在了。
    LANG_NAMES = ("英语", "英文", "日语", "日文", "韩语", "韩文",
                  "俄语", "法语", "德语", "西班牙语", "意大利语", "泰语", "阿拉伯语")
    HIGH_PRIORITY = [
        # 翻译类："X用英语怎么说" / "怎么翻译成英文" / "教我英语"
        (r"(?:用|用.{0,4}?)(?:%s).{0,4}(?:怎么|如何).{0,3}(?:说|讲|读|写|表达)" % "|".join(LANG_NAMES),
         "lang_learning"),
        (r"(?:翻译|译).{0,6}(?:%s)" % "|".join(LANG_NAMES), "lang_learning"),
        (r"(?:%s).{0,4}(?:怎么说|怎么讲|怎么读|怎么写|是什么|怎么说呢)" % "|".join(LANG_NAMES),
         "lang_learning"),
        (r"^(?:教我|我想学|学习|学)(?:%s|外语|单词)" % "|".join(LANG_NAMES), "lang_learning"),
        (r"(?:外语|单词|词汇|语法).{0,3}(?:怎么|如何).{0,3}(?:学|记|背)", "lang_learning"),
        # 生僻词释义类：词 + 释义动词
        (r"(?:什么意思|啥意思|啥{0,2}意思|怎么读|怎么念|念什么|啥{0,2}字|是啥{0,2}意思|典故|出处)",
         "rare_word"),
        (r"^(?:查|查一下|查查|解释|介绍|说说).{0,8}(?:这个词|这个字|什么意思)", "rare_word"),
    ]
    # 生僻字特征：CJK 扩展区 + 康熙部首生僻字 + 罕见字块（CJK 扩展 B 起）
    RARE_CHAR = r"[\u3400-\u4dbf\uf900-\ufaff\U00020000-\U0002a6df]"

    # 短句停用词：这些是高频口语但没有关键词命中，不能当成"生僻字"（如"我不要"）
    SHORT_STOPWORDS = {
        "我", "你", "他", "她", "它", "是", "不", "好", "嗯", "啊", "吧", "呢", "哦", "哈",
        "算了", "不要", "不是", "好吧", "没事", "对吧", "是吗", "怎样", "如何", "多少",
    }

    PATTERN_RULES = [
        (r"^密码\s+", "admin_cmd"), (r"^激活\s+", "admin_cmd"),
        (r"^调\w+值\s+\d+", "admin_cmd"), (r"^写日记\s+", "admin_cmd"),
        (r"^教我说话\s+", "admin_cmd"), (r"^预览\s+", "admin_cmd"),
        (r"^重置学习\s*", "admin_cmd"), (r"^换性格\s+", "admin_cmd"),
        (r"^写(.*?)(代码|程序)", "action_request"),
        (r"^(1\+1|\d+\+\d+|等于几|是多少)", "question"),
        (r"^(你会|你能|你可以).*?(吗|么|嘛)", "question"),
        (r"^(如果|假如).*?(怎么办|会怎样)", "question"),
        (r"^(我|人家|人家想).*?(你|安晚).*?(吗|嘛|么)$", "state_query"),
        (r"^(别|不要|不许).*?(生气|难过|哭|走|离开)", "action_request"),
        (r"^(原谅|我错了|对不起).*?我", "action_request"),
    ]

    @classmethod
    def label(cls, question):
        import re as _re

        q = question.strip()
        if q in cls.EXACT_RULES:
            return cls.EXACT_RULES[q], 1.0
        # 高优先级：语言学习 / 生僻词释义（先于关键词表，避免被 question 抢走）
        for pattern, intent in cls.HIGH_PRIORITY:
            if _re.search(pattern, q):
                return intent, 0.9
        # 跨所有意图找**最长**匹配关键词。原实现按 dict 顺序第一个命中就返回，
        # 于是 "你好笨" 会先被 greeting 的 "你好" 抢走 —— 长关键词被短关键词掩盖，
        # 标成问候而不是调侃，意图头跟着学错。
        # 注意比较必须用 len(kw)：置信度分数 min(len/len(q)+0.3, .95) 会被截断到
        # 同一个 0.95，短词反而先赢。长度才是"匹配得更具体"的真实信号。
        best_intent, best_len, best_kw = None, 0, None
        for intent, keywords in cls.KEYWORD_RULES.items():
            for kw in keywords:
                if kw in q and len(kw) > best_len:
                    best_intent, best_len, best_kw = intent, len(kw), kw
        if best_kw:
            return best_intent, min(best_len / max(len(q), 1) + 0.3, 0.95)
        for pattern, intent in cls.PATTERN_RULES:
            if _re.search(pattern, q):
                return intent, 0.85
        # 生僻字兜底：整句就是 1~3 个汉字（无标点、无助词）→ 查词意图。
        # 不能靠码位范围判断"生僻"：Unicode 不标生僻，龘(U+9F98)、齾(U+9FFE)
        # 都在 CJK 基本区 4E00-9FFF 里。放最后兜底，保证"你好""喵"这类
        # 能被关键词表先认领。
        if (len(q) <= 2 and
                re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]{1,2}", q)
                and q not in cls.SHORT_STOPWORDS):
            return "rare_word", 0.75
        return "chat", 0.5

    @classmethod
    def to_index(cls, intent_name):
        return cls.INTENT2IDX.get(intent_name, cls.INTENT2IDX["chat"])


# ==================== 内嵌数据（已移除：不再内嵌权重/词表/语料，从零训练） ====================


# ==================== 数据管线（原 anwan/data_pipeline.py） ====================

def load_builtin_corpus():
    """加载内置语料：data_raw 字面量 + 话语树样本 + 对比对，按 question 去重"""
    _raw = [("你好", ["你好喵~"]), ("在吗", ["在呢在呢喵~"]), ("再见", ["再见喵~"]), ("你是谁", ["我是安晚喵~"]), ("你喜欢什么", ["安晚喜欢主人喵~"])]

    pairs = list(_raw)
    existing = {q for q, _ in pairs}

    # 话语树样本：模拟多轮对话流（与旧版 generate_discourse_tree_samples 一致）
    discourse_samples = [
        ("你好", ["哼，现在才来找安晚喵！"]),
        ("在干嘛", ["在想怎么怼你，满意了吗喵"]),
        ("这么凶", ["安晚一直很凶的喵~（得意）"]),
        ("我心情不好", ["（蹭蹭）别难过了喵~"]),
        ("还是难过", ["安晚陪你，不许不开心喵~"]),
        ("谢谢你", ["哼...安晚才不是专门安慰你呢喵~"]),
        ("你叫什么", ["安晚！记不住就咬你喵~"]),
        ("安晚你好", ["（歪头）你刚才不是问过了吗喵？"]),
        ("我忘了", ["（叹气）安晚叫安晚啦！记好了喵~"]),
        ("我喜欢你", ["（脸红）安晚也...也喜欢你啦喵♡"]),
        ("骗你的", ["（炸毛）你...你太过分了喵！(｀Д´)"]),
        ("对不起", ["（扭头）...那你要哄好安晚才行喵~"]),
    ]
    for q, a in discourse_samples:
        if q not in existing:
            pairs.append((q, a))
            existing.add(q)

    # 最小对比对：相似输入 → 相反意图（与旧版 generate_contrastive_pairs 一致）
    contrast_groups = [
        ("我喜欢你", "我讨厌你", "（脸红）安晚也...也喜欢你啦喵♡", "（炸毛）安晚才...才不在意喵！"),
        ("你好可爱", "你好笨", "（脸红）哼...算你有眼光喵~", "？？？你再说一遍试试喵"),
        ("我开心了", "我难过了", "（开心摇尾巴）太好了喵~♡", "（蹭蹭）别难过了喵~"),
        ("爱你", "恨你", "（脸通红）安晚...安晚也喜欢主人啦！♡", "（低头）...安晚知道了喵..."),
        ("抱抱", "滚开", "（扑过去）抱到了喵~♡", "（炸毛）你...你才滚开喵！(｀Д´)"),
        ("你好", "再见", "哼，现在才来找安晚喵！", "快滚吧，明天再来喵~"),
        ("早安", "晚安", "（揉眼睛）这么早...你是魔鬼吗喵", "（蜷成团）晚安~"),
        ("我成功了", "我失败了", "（跳起来）太棒了！主人最厉害喵~♡", "失败算什么，安晚陪你重新来喵~"),
    ]
    for pos_q, neg_q, pos_a, neg_a in contrast_groups:
        for q, a in ((pos_q, pos_a), (neg_q, neg_a)):
            if q not in existing:
                pairs.append((q, [a]))
                existing.add(q)
    return pairs


# ---------------- 通用语料读取 ----------------
# 原实现只扫 .txt/.json/.md，且 _parse_json 只认 q/question + a/answer。
# 真实语料的格式远不止这些：JSONL、Alpaca(instruction/output)、ShareGPT(from/value)、
# OpenAI messages(role/content)、CSV/TSV、对话文本、纯散文……全部会被静默丢成 []，
# 用户以为在训练，其实一条都没进去。这里做格式自动识别 + 失败必报警。
Q_KEYS = ("q", "Q", "question", "query", "prompt", "instruction", "input",
          "human", "问题", "问", "user", "user_message")
A_KEYS = ("a", "A", "answer", "response", "output", "completion", "result",
          "gpt", "bot", "assistant", "答案", "答", "reply")
# 注意：messages / conversations 故意不在这里。它们是**对话序列**，得留给
# _obj_to_pair 去配对"第1条 user + 第2条 assistant"；在顶层拆开后列表里只剩
# 单条消息，反倒配不出问答对（实测 OpenAI messages 格式就是这么丢的）。
WRAP_KEYS = ("data", "dataset", "train", "samples", "rows", "records", "items", "list")


def _read_text_any(fpath):
    """按 utf-8 → gbk → 忽略错误 三级回退读取，永远不抛异常"""
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            with open(fpath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            return ""
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


_ROLE_VALUES = {"user", "assistant", "system", "human", "gpt", "bot",
                "user_message", "assistant_message", "q", "a", ""}


def _salvage_texts(d):
    """从非问答对象里抢救可用文本：过滤角色名等噪声值"""
    out = []
    for k, v in d.items():
        if not isinstance(v, str):
            continue
        t = v.strip()
        if len(t) < 8:                       # 太短的基本是角色名/标记
            continue
        if t.lower() in _ROLE_VALUES:
            continue
        out.append(t)
    return out


def _pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float)):
                return str(v)
    return None


def _obj_to_pair(d):
    """单个对象 → (问, 答) 或 None"""
    if not isinstance(d, dict):
        return None
    # OpenAI/ShareGPT 风格：role/content 或 from/value 的连续对话
    for k in ("messages", "conversations", "conversation", "dialog", "turns"):
        if isinstance(d.get(k), list) and d[k] and isinstance(d[k][0], dict):
            seq = d[k]
            role = seq[0].get("role") or seq[0].get("from") or ""
            val = seq[0].get("content") or seq[0].get("value") or seq[0].get("text") or ""
            for nxt in seq[1:]:
                nrole = nxt.get("role") or nxt.get("from") or ""
                nval = nxt.get("content") or nxt.get("value") or nxt.get("text") or ""
                if str(nrole).lower() in ("assistant", "gpt", "bot", "ai") and \
                   str(role).lower() in ("user", "human"):
                    return str(val).strip(), str(nval).strip()
                role, val = nrole, nval
    q = _pick(d, Q_KEYS)
    a = _pick(d, A_KEYS)
    if q and a:
        return q, a
    return None


def _parse_json_generic(raw):
    """吃下 JSON 数组 / JSONL / 带包装层的 dict / 单对象 / Alpaca / ShareGPT / OpenAI"""
    pairs, texts = [], []
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        # JSONL：逐行解析
        ok = 0
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ok += 1
            pr = _obj_to_pair(obj)
            if pr:
                pairs.append(pr)
            elif isinstance(obj, dict):
                texts.extend(_salvage_texts(obj))
            elif isinstance(obj, str) and len(obj) > 2:
                texts.append(obj)
        if ok:
            return pairs, texts, "jsonl"
        return [], [], "失败:JSON/JSONL 都解析不了"
    if isinstance(data, dict):
        for k in WRAP_KEYS:
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            pr = _obj_to_pair(data)
            return ([pr] if pr else []), texts, ("json-obj" if pr else "json-obj无问答字段")
    if isinstance(data, list):
        for item in data:
            pr = _obj_to_pair(item) if isinstance(item, (dict, list)) else None
            if pr:
                pairs.append(pr)
            elif isinstance(item, str) and len(item) > 2:
                texts.append(item)
            elif isinstance(item, dict):
                texts.extend(_salvage_texts(item))
    return pairs, texts, ("json" if pairs else "json-仅散文")


def _parse_csv_generic(raw, delim=","):
    import csv as _csv
    import io as _io
    pairs, texts = [], []
    try:
        rows = list(_csv.reader(_io.StringIO(raw), delimiter=delim))
    except Exception:
        return [], [], "失败:CSV 解析异常"
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [], [], "空"
    hdr = [c.strip().lower() for c in rows[0]]
    has_hdr = any(h in Q_KEYS or h in A_KEYS for h in hdr)
    if has_hdr:
        qi = next((i for i, h in enumerate(hdr) if h in Q_KEYS), 0)
        ai = next((i for i, h in enumerate(hdr) if h in A_KEYS), 1 if len(hdr) > 1 else 0)
        body = rows[1:]
    else:
        qi, ai, body = 0, (1 if rows[0] and len(rows[0]) > 1 else 0), rows
    for r in body:
        if max(qi, ai) >= len(r):
            continue
        q, a = r[qi].strip(), r[ai].strip()
        if q and a:
            pairs.append((q, a))
    return pairs, texts, ("csv" if pairs else "csv-无有效行")


def _parse_txt_generic(raw):
    """显式标记 → 行对启发式 → 段落对 → 纯散文（交给 LM 阶段）"""
    import re as _r
    pairs, texts = [], []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    qm = _r.compile(r"^(?:Q|问|问题|User|用户|You)\s*[:：]\s*(.+)$", _r.I)
    am = _r.compile(r"^(?:A|答|答案|Assistant|Bot|安晚|GPT)\s*[:：]\s*(.+)$", _r.I)
    i = 0
    while i < len(lines) - 1:
        mq, ma = qm.match(lines[i]), am.match(lines[i + 1])
        if mq and ma:
            pairs.append((mq.group(1).strip(), ma.group(1).strip()))
            i += 2
            continue
        i += 1
    if not pairs:
        # 行对启发式。关键：问句不能以句号/叹号收尾 —— 那是陈述句（散文、
        # 百科、说明文），不是提问。允许无标点（"在吗"）或以问号收尾。
        for j in range(0, len(lines) - 1, 2):
            q, a = lines[j], lines[j + 1]
            if len(q) > 60 or q.endswith(("。", "！", "!", "；", ";")):
                continue
            if len(a) > len(q) or any(ch in a for ch in "。！？喵~"):
                pairs.append((q, a))
    if not pairs:
        # 不做段落配对：纯散文（说明文/百科/小说）的两段之间没有问答关系，
        # 硬配会把"第1段"和"第2段"当成一问一答，污染 SFT 语料。
        # 段落配对只保留在 _parse_md（那里确实是对话记录格式）。
        texts = [l for l in lines if len(l) > 8]
    return pairs, texts, ("txt" if pairs else "txt-纯散文")


def parse_corpus_file(fpath):
    """自动识别格式并解析。返回 (pairs, raw_texts, format_name) —— 绝不静默失败。"""
    ext = os.path.splitext(fpath)[1].lower()
    raw = _read_text_any(fpath)
    if not raw.strip():
        return [], [], "空文件"
    if ext in (".jsonl", ".ndjson") or (raw.count("\n{") > 2 and not raw.lstrip().startswith("[")):
        return _parse_json_generic(raw)
    if ext == ".json" or raw.lstrip()[:1] in ("{", "["):
        return _parse_json_generic(raw)
    if ext == ".csv" or (ext == "" and raw.count(",") > raw.count("\n") * 2):
        return _parse_csv_generic(raw, ",")
    if ext == ".tsv":
        return _parse_csv_generic(raw, "\t")
    if ext in (".txt", ".md", ".text", ".log", "") or True:
        return _parse_txt_generic(raw)


def scan_corpus(folder_path, verbose=True):
    """递归扫描目录下的所有语料文件，返回 (pairs, raw_texts, 报告行列表)。
    报告会把"解析不出来"的文件单独列出来 —— 这是最容易被忽略的坑。"""
    pairs, texts, report, skipped = [], [], [], []
    if not os.path.isdir(folder_path):
        return [], [], [f"[{folder_path}] 目录不存在"], []
    n_files = 0
    for root, _dirs, files in os.walk(folder_path):
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".txt", ".json", ".jsonl", ".ndjson", ".md", ".csv",
                           ".tsv", ".text", ".log", ""):
                skipped.append((fn, f"扩展名 {ext} 不支持"))
                continue
            fp = os.path.join(root, fn)
            if not os.path.isfile(fp):
                continue
            n_files += 1
            try:
                p, t, fmt = parse_corpus_file(fp)
            except Exception as e:                     # 任何异常都变成报告，不吞掉
                skipped.append((fn, f"异常 {type(e).__name__}: {e}"))
                continue
            pairs.extend(p)
            texts.extend(t)
            rel = os.path.relpath(fp, folder_path)
            if p or t:
                report.append(f"    {rel}  [{fmt}]  问答{len(p)} 原文{len(t)}")
            else:
                skipped.append((rel, f"未识别到任何内容（{fmt}）"))
    if verbose:
        print(f"[语料扫描] {folder_path}")
        print(f"    文件 {n_files} 个 | 问答 {len(pairs)} 条 | 原文 {len(texts)} 段")
        for line in report[:20]:
            print(line)
        if len(report) > 20:
            print(f"    …… 另有 {len(report)-20} 个文件")
        if skipped:
            print(f"    [!] {len(skipped)} 个文件没能解析（这些语料不会被使用）：")
            for fn, why in skipped[:10]:
                print(f"        {fn} — {why}")
            if len(skipped) > 10:
                print(f"        …… 另有 {len(skipped)-10} 个")
    return pairs, texts, report, skipped


def load_folder_pairs(folder_path):
    """兼容旧接口：只要问答对。推荐改用 scan_corpus() 拿到完整报告。"""
    pairs, _texts, _rep, _skip = scan_corpus(folder_path, verbose=False)
    return pairs


def _parse_txt(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(fpath, "r", encoding="gbk") as f:
                content = f.read()
        except Exception:
            return []
    pairs = []
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("Q:", "问：", "Q：")):
            q = line[2:].strip()
            if i + 1 < len(lines) and lines[i + 1].startswith(("A:", "答：", "A：")):
                pairs.append((q, lines[i + 1][2:].strip()))
                i += 2
                continue
        elif i + 1 < len(lines) and len(line) <= 50:
            q, a = line, lines[i + 1]
            if len(a) > len(q) or any(ch in a for ch in "。！？喵"):
                pairs.append((q, a))
                i += 2
                continue
        i += 1
    if not pairs and len(lines) >= 2:
        for j in range(0, len(lines) - 1, 2):
            pairs.append((lines[j], lines[j + 1]))
    return pairs


def _parse_json(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    pairs = []
    entries = data if isinstance(data, list) else data.get("entries", []) if isinstance(data, dict) else []
    if isinstance(data, dict) and not data.get("entries") and ("q" in data or "a" in data):
        entries = [data]
    for item in entries:
        if isinstance(item, dict):
            q = item.get("q") or item.get("question") or item.get("Q")
            a = item.get("a") or item.get("answer") or item.get("A")
            if q and a:
                pairs.append((str(q), str(a)))
    return pairs


def _parse_md(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []
    pairs = []
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith((">", "-", "*")):
            clean = line.lstrip(">-* ").strip()
            if i + 1 < len(lines):
                next_line = lines[i + 1].lstrip(">-* ").strip()
                if len(clean) <= 50 and len(next_line) > 0:
                    pairs.append((clean, next_line))
                    i += 2
                    continue
        i += 1
    if not pairs:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for j in range(0, len(paragraphs) - 1, 2):
            pairs.append((paragraphs[j], paragraphs[j + 1]))
    return pairs


class DataPipeline:
    """语料 → 训练批次（LM 阶段 / SFT 阶段）"""

    def __init__(self, tokenizer, max_len=None, folder_path=None, verbose=True):
        self.tokenizer = tokenizer
        self.max_len = max_len or config.MAX_LEN
        self.qa_pairs = load_builtin_corpus()
        folder = folder_path or config.TRAIN_FOLDER
        # 通用扫描：支持 JSONL/Alpaca/ShareGPT/OpenAI/CSV/TSV/对话文本/散文，
        # 并把解析不出来的文件单独报告出来（原来这些会静默变成 0 条）
        ext_pairs, raw_texts, _rep, _skip = scan_corpus(folder, verbose=verbose)
        self.raw_texts = raw_texts          # 无问答结构的散文 → 只进 LM 阶段
        self.parse_report = _rep
        self.parse_skipped = _skip
        if ext_pairs:
            existing = {q for q, _ in self.qa_pairs}
            added = 0
            for q, a in ext_pairs:
                if q not in existing:
                    self.qa_pairs.append((q, [a]))
                    existing.add(q)
                    added += 1
            print(f"  [folder] 外部语料 {folder}: 新增 {added} 条问答")
        if self.raw_texts:
            print(f"  [folder] 另有 {len(self.raw_texts)} 段无问答结构的原文，"
                  f"只用于 LM 预训练")
        self.texts = []
        for q, answers in self.qa_pairs:
            self.texts.append(q)
            if isinstance(answers, list):
                self.texts.extend(answers)
            else:
                self.texts.append(answers)
        self.texts.extend(self.raw_texts)

    # ---------------- 语料统计 ----------------
    def stats(self):
        n_ans = sum(len(a) if isinstance(a, list) else 1 for _, a in self.qa_pairs)
        return {"qa": len(self.qa_pairs), "answers": n_ans, "texts": len(self.texts),
                "raw_texts": len(getattr(self, "raw_texts", [])),
                "files": len(getattr(self, "parse_report", [])),
                "skipped": len(getattr(self, "parse_skipped", []))}

    # ---------------- SFT 批次（阶段二） ----------------
    def sft_batches(self, batch_size, device, shuffle=True, seed=0):
        """问答对 teacher forcing 批次；附带一体化意图标签；按 MULTI_TURN_PROB 概率掺入多轮历史样本"""
        import random

        rng = random.Random(seed)
        items = []
        for q, answers in self.qa_pairs:
            ans_list = answers if isinstance(answers, list) else [answers]
            for a in ans_list:
                items.append((q, a))
        if shuffle:
            rng.shuffle(items)

        corpus = [(q, a) for q, answers in self.qa_pairs
                  for a in (answers if isinstance(answers, list) else [answers])]

        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            srcs, tgt_ins, tgt_outs, intents = [], [], [], []
            for q, a in chunk:
                use_history = rng.random() < config.MULTI_TURN_PROB and len(corpus) > 1
                pack_prob = getattr(config, "CONTENT_PACK_PROB", 0.0)
                if use_history:
                    n_prev = rng.randint(1, min(config.HISTORY_TURNS, len(corpus) - 1))
                    history = rng.sample(corpus, n_prev)
                    src_idx = build_multi_turn_src(
                        self.tokenizer, history, q,
                        max_len=self.max_len, history_turns=None)
                elif pack_prob and rng.random() < pack_prob and len(self.texts) > 1:
                    n_ctx = rng.randint(1, max(1, getattr(config, "CONTENT_PACK_MAX_SEGS", 4) - 1))
                    ctx = rng.sample(self.texts, min(n_ctx, len(self.texts)))
                    src_idx = pack_content_segments(
                        self.tokenizer, list(ctx) + [q], max_len=self.max_len)
                else:
                    q_ids = self.tokenizer.encode(q, max_len=self.max_len - 1)
                    src_idx = [config.SPECIAL_TOKENS["<BOS>"]] + q_ids
                a_ids = self.tokenizer.encode(a, max_len=self.max_len - 1)
                tgt_in = [config.SPECIAL_TOKENS["<BOS>"]] + a_ids
                tgt_out = a_ids + [config.SPECIAL_TOKENS["<EOS>"]]
                srcs.append(src_idx[: self.max_len])
                tgt_ins.append(tgt_in[: self.max_len] + [config.SPECIAL_TOKENS["<PAD>"]] * max(0, self.max_len - len(tgt_in)))
                tgt_outs.append(tgt_out[: self.max_len] + [config.SPECIAL_TOKENS["<PAD>"]] * max(0, self.max_len - len(tgt_out)))

                intent_name, _ = UnifiedIntentLabeler.label(q)
                intents.append(UnifiedIntentLabeler.to_index(intent_name))
            T = max(len(s) for s in srcs)
            src_padded = [s + [config.SPECIAL_TOKENS["<PAD>"]] * (T - len(s)) for s in srcs]
            yield {
                "src": torch.tensor(src_padded, dtype=torch.long, device=device),
                "tgt_in": torch.tensor(tgt_ins, dtype=torch.long, device=device),
                "tgt_out": torch.tensor(tgt_outs, dtype=torch.long, device=device),
                "intents": torch.tensor(intents, dtype=torch.long, device=device),
            }

    # ---------------- LM 批次（阶段一） ----------------
    def lm_batches(self, batch_size, device, seed=0):
        """每条语料文本自监督 next-token：src=文本、tgt=文本自身（学语言模式）"""
        import random

        rng = random.Random(seed)
        texts = list(self.texts)
        rng.shuffle(texts)

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            srcs, tgt_ins, tgt_outs = [], [], []
            pack_prob = getattr(config, "CONTENT_PACK_PROB", 0.0)
            max_segs = getattr(config, "CONTENT_PACK_MAX_SEGS", 4)
            for t in chunk:
                t_ids = self.tokenizer.encode(t, max_len=self.max_len - 1)
                if pack_prob and rng.random() < pack_prob and len(texts) > 1:
                    n_ctx = rng.randint(1, max(1, max_segs - 1))
                    extra = rng.sample(texts, min(n_ctx, len(texts)))
                    src_idx = pack_content_segments(
                        self.tokenizer, list(extra) + [t], max_len=self.max_len)
                else:
                    src_idx = [config.SPECIAL_TOKENS["<BOS>"]] + t_ids
                tgt_in = [config.SPECIAL_TOKENS["<BOS>"]] + t_ids
                tgt_out = t_ids + [config.SPECIAL_TOKENS["<EOS>"]]
                srcs.append(src_idx[: self.max_len])
                # 长度对齐（同一句内 tgt_in == tgt_out 长度）
                tgt_in = tgt_in[: self.max_len]
                tgt_out = tgt_out[: self.max_len]
                while len(tgt_in) < len(tgt_out):
                    tgt_in.append(config.SPECIAL_TOKENS["<PAD>"])
                while len(tgt_out) < len(tgt_in):
                    tgt_out.append(config.SPECIAL_TOKENS["<PAD>"])
                tgt_ins.append(tgt_in)
                tgt_outs.append(tgt_out)
            T = max(len(s) for s in srcs)
            T2 = max(len(t) for t in tgt_ins)
            src_padded = [s + [config.SPECIAL_TOKENS["<PAD>"]] * (T - len(s)) for s in srcs]
            tgt_in_p = [t + [config.SPECIAL_TOKENS["<PAD>"]] * (T2 - len(t)) for t in tgt_ins]
            tgt_out_p = [t + [config.SPECIAL_TOKENS["<PAD>"]] * (T2 - len(t)) for t in tgt_outs]
            yield {
                "src": torch.tensor(src_padded, dtype=torch.long, device=device),
                "tgt_in": torch.tensor(tgt_in_p, dtype=torch.long, device=device),
                "tgt_out": torch.tensor(tgt_out_p, dtype=torch.long, device=device),
                "intents": None,
            }


# ==================== 训练数据文件夹系统（原老版 A10 TrainingDataFolder） ====================

class TrainingDataFolder:
    """
    训练数据文件夹系统：让安晚"慢慢"阅读学习文件夹里的文本
    支持渐进式学习、进度持久化、多种文件格式
    """
    def __init__(self, folder_name=LEARN_FOLDER_NAME, base_dir=None):
        self.base_dir = base_dir or BASE_DIR
        self.folder_path = os.path.join(self.base_dir, folder_name)
        self.progress_file = os.path.join(self.base_dir, PROGRESS_FILE_NAME)
        self.supported_exts = {'.txt', '.json', '.md'}
        self.progress = self._load_progress()
        self.new_learned = []
        self.max_learn_per_session = 50
        os.makedirs(self.folder_path, exist_ok=True)
        self.scanned_files = self._scan_files()

    def _scan_files(self):
        files = []
        if not os.path.exists(self.folder_path):
            return files
        for fname in sorted(os.listdir(self.folder_path)):
            if fname == LEARNED_PAIRS_NAME:       # 跳过学习成果落盘文件自身，避免循环学习
                continue
            fpath = os.path.join(self.folder_path, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.supported_exts:
                    files.append(fpath)
        return files

    def _load_progress(self):
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def _save_progress(self):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def _parse_txt(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            try:
                with open(fpath, 'r', encoding='gbk') as f:
                    content = f.read()
            except:
                return []
        qa_pairs = []
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('Q:') or line.startswith('问：') or line.startswith('Q：'):
                q = line[2:].strip()
                if i + 1 < len(lines):
                    a_line = lines[i + 1]
                    if a_line.startswith('A:') or a_line.startswith('答：') or a_line.startswith('A：'):
                        qa_pairs.append((q, a_line[2:].strip()))
                        i += 2
                        continue
            elif i + 1 < len(lines) and len(line) <= 50:
                q, a = line, lines[i + 1]
                if len(a) > len(q) or '。' in a or '！' in a or '？' in a or '喵' in a:
                    qa_pairs.append((q, a))
                    i += 2
                    continue
            i += 1
        if not qa_pairs and len(lines) >= 2:
            for j in range(0, len(lines) - 1, 2):
                qa_pairs.append((lines[j], lines[j + 1]))
        return qa_pairs

    def _parse_json(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return []
        qa_pairs = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    q = item.get('q') or item.get('question') or item.get('Q')
                    a = item.get('a') or item.get('answer') or item.get('A')
                    if q and a:
                        qa_pairs.append((str(q), str(a)))
        elif isinstance(data, dict):
            entries = data.get('entries', [])
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict):
                        q = item.get('q') or item.get('question')
                        a = item.get('a') or item.get('answer')
                        if q and a:
                            qa_pairs.append((str(q), str(a)))
        return qa_pairs

    def _parse_md(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return []
        qa_pairs = []
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('>') or line.startswith('-') or line.startswith('*'):
                clean = line.lstrip('>-* ').strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1].lstrip('>-* ').strip()
                    if len(clean) <= 50 and len(next_line) > 0:
                        qa_pairs.append((clean, next_line))
                        i += 2
                        continue
            i += 1
        if not qa_pairs:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for j in range(0, len(paragraphs) - 1, 2):
                qa_pairs.append((paragraphs[j], paragraphs[j + 1]))
        return qa_pairs

    def _parse_file(self, fpath):
        ext = os.path.splitext(fpath)[1].lower()
        if ext == '.json':
            return self._parse_json(fpath)
        elif ext == '.md':
            return self._parse_md(fpath)
        else:
            return self._parse_txt(fpath)

    def _is_valid_qa(self, q, a):
        if not q or not a:
            return False
        if len(q) < 2 or len(a) < 2:
            return False
        if re.match(r'^[^\w\u4e00-\u9fff]+$', q) or re.match(r'^[^\w\u4e00-\u9fff]+$', a):
            return False
        if q == a:
            return False
        return True

    def learn(self):
        if not self.scanned_files:
            return [], "[folder] 训练数据文件夹为空，请在文件夹里放 .txt/.json/.md 文件"
        all_new_pairs = []
        stats = {'files_scanned': 0, 'files_completed': 0, 'files_in_progress': 0, 'skipped_files': 0}
        for fpath in self.scanned_files:
            fname = os.path.basename(fpath)
            rel_path = os.path.relpath(fpath, self.base_dir)
            prog = self.progress.get(rel_path, {})
            if prog.get('completed', False):
                stats['files_completed'] += 1
                continue
            qa_pairs = self._parse_file(fpath)
            if not qa_pairs:
                stats['skipped_files'] += 1
                self.progress[rel_path] = {'completed': True, 'reason': 'no_valid_qa'}
                continue
            total = len(qa_pairs)
            learned = prog.get('lines_learned', 0)
            remaining = total - learned
            if remaining <= 0:
                self.progress[rel_path] = {'lines_learned': total, 'total_lines': total, 'completed': True}
                stats['files_completed'] += 1
                continue
            batch_size = min(remaining, self.max_learn_per_session - len(all_new_pairs))
            if batch_size <= 0:
                break
            batch = qa_pairs[learned:learned + batch_size]
            valid_batch = [(q, a) for q, a in batch if self._is_valid_qa(q, a)]
            all_new_pairs.extend(valid_batch)
            new_learned = learned + len(batch)
            self.progress[rel_path] = {
                'lines_learned': new_learned,
                'total_lines': total,
                'completed': new_learned >= total
            }
            stats['files_scanned'] += 1
            if new_learned >= total:
                stats['files_completed'] += 1
            else:
                stats['files_in_progress'] += 1
            if len(all_new_pairs) >= self.max_learn_per_session:
                break
        self.new_learned = all_new_pairs
        self._save_progress()
        if all_new_pairs:
            self.export_to_corpus(all_new_pairs)
        info = f"(｡･ω･｡)ﾉ 训练数据学习报告\n"
        info += f"  扫描文件: {len(self.scanned_files)} 个\n"
        info += f"  本次学习: {len(all_new_pairs)} 条新数据\n"
        info += f"  已完成: {stats['files_completed']} 个文件\n"
        info += f"  学习中: {stats['files_in_progress']} 个文件\n"
        if all_new_pairs:
            info += f"  来源示例: {os.path.basename(self.scanned_files[0]) if self.scanned_files else 'N/A'}\n"
        return all_new_pairs, info

    def export_to_corpus(self, new_pairs):
        """把本次学到的配对合并写入 train_data/learned_pairs.json（按问句去重），
        DataPipeline 读取 TRAIN_FOLDER 时统一并入，实现 learn→train→chat 数据闭环。"""
        merged = {}
        if os.path.isfile(LEARNED_PAIRS_FILE):
            try:
                with open(LEARNED_PAIRS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for it in data:
                        if isinstance(it, dict) and it.get("q") and it.get("a"):
                            merged[str(it["q"])] = str(it["a"])
            except Exception:
                merged = {}
        added = 0
        for _q, _a in new_pairs:
            if _q not in merged:
                merged[str(_q)] = str(_a)
                added += 1
        os.makedirs(os.path.dirname(LEARNED_PAIRS_FILE), exist_ok=True)
        with open(LEARNED_PAIRS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"q": k, "a": v} for k, v in merged.items()],
                      f, ensure_ascii=False, indent=2)
        print(f"  [闭环] 学习成果已并入训练语料: {LEARNED_PAIRS_FILE}（本次新增 {added}，累计 {len(merged)} 条）")
        return added

    def get_stats(self):
        total_files = len(self.scanned_files)
        completed = sum(1 for p in self.progress.values() if p.get('completed', False))
        total_learned = sum(p.get('lines_learned', 0) for p in self.progress.values())
        total_lines = sum(p.get('total_lines', 0) for p in self.progress.values())
        pct = total_learned / total_lines * 100 if total_lines > 0 else 0
        lines = [
            "(｡･ω･｡)ﾉ 训练数据文件夹统计",
            f"  文件夹: {self.folder_path}",
            f"  文件数: {total_files}",
            f"  已完成: {completed}/{total_files}",
            f"  学习进度: {total_learned}/{total_lines} ({pct:.1f}%)",
        ]
        if self.scanned_files:
            lines.append("  文件详情:")
            for fpath in self.scanned_files:
                fname = os.path.basename(fpath)
                rel_path = os.path.relpath(fpath, self.base_dir)
                prog = self.progress.get(rel_path, {})
                if prog.get('completed', False):
                    status = "[OK] 已完成"
                elif prog.get('lines_learned', 0) > 0:
                    status = f"↻ {prog['lines_learned']}/{prog.get('total_lines', '?')}"
                else:
                    status = "⏳ 未开始"
                lines.append(f"    {fname}: {status}")
        return '\n'.join(lines)

    def force_relearn(self, filename=None):
        if filename:
            for fpath in self.scanned_files:
                if os.path.basename(fpath) == filename:
                    rel_path = os.path.relpath(fpath, self.base_dir)
                    if rel_path in self.progress:
                        del self.progress[rel_path]
                    return f"↻ 已重置「{filename}」的学习进度"
            return f"❌ 找不到文件「{filename}」"
        else:
            self.progress = {}
            self._save_progress()
            return "↻ 已重置所有学习进度，下次启动将重新学习"

    def peek_file(self, filename):
        for fpath in self.scanned_files:
            if os.path.basename(fpath) == filename:
                qa_pairs = self._parse_file(fpath)
                preview = []
                for i, (q, a) in enumerate(qa_pairs[:5]):
                    preview.append(f"  [{i+1}] Q: {q[:30]}...")
                    preview.append(f"      A: {a[:30]}...")
                if len(qa_pairs) > 5:
                    preview.append(f"  ... 共 {len(qa_pairs)} 条")
                return '\n'.join(preview)
        return f"❌ 找不到文件「{filename}」"



# ==================== 评测（原 anwan/evaluate.py） ====================

def _char_overlap(pred, ref):
    """字符级 Jaccard / F1（按字符集合）"""
    p = set(pred)
    r = set(ref)
    if not r:
        return 1.0 if not p else 0.0
    inter = len(p & r)
    if inter == 0:
        return 0.0
    prec = inter / max(len(p), 1)
    rec = inter / len(r)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate_generation(model, tokenizer, pipeline, n=20, device="cpu"):
    """生成质量：对前 n 条问答对（含外部）贪婪生成并计算指标"""
    model.eval()
    pairs = pipeline.qa_pairs[:n]
    f1s, hits, lens, empties = [], 0, 0, 0
    samples = []
    for q, answers in pairs:
        ref = (answers[0] if isinstance(answers, list) else answers) or ""
        q_ids = tokenizer.encode(q, max_len=config.MAX_LEN - 1)
        src = torch.tensor(
            [[config.SPECIAL_TOKENS["<BOS>"]] + q_ids], dtype=torch.long, device=device
        )
        gen, _, _, _, _ = model.generate(
            src, max_len=config.GEN_MAX_LEN, temperature=0.8,
            top_p=config.TOP_P, strategy="greedy",
        )
        resp = tokenizer.decode(gen[0].cpu().tolist()).strip()
        lens += len(resp)
        if len(resp) < config.REPLY_MIN_LEN:
            empties += 1
        else:
            f1 = _char_overlap(resp, ref)
            f1s.append(f1)
            if f1 >= 0.35:
                hits += 1
        if len(samples) < 5:
            samples.append((q, resp, ref))
    n_valid = max(len(f1s), 1)
    return {
        "f1_mean": sum(f1s) / n_valid,
        "hit_rate@0.35": hits / max(len(pairs), 1),
        "avg_len": lens / max(len(pairs), 1),
        "empty_rate": empties / max(len(pairs), 1),
        "samples": samples,
    }


@torch.no_grad()
def evaluate_router(model, tokenizer, texts=None, n=30, repeats=3, device="cpu"):
    """
    分区稳定性：
    - 边际路由熵（batch 平均 router 分布熵，越低越坍缩）
    - 四模块平均载荷（自动分区倾向）
    - 同输入多次生成的权重方差（稳定性）
    """

    model.eval()
    if texts is None:
        texts = [q for q, _ in load_builtin_corpus()[:n]]

    weights_sum = None
    count = 0
    repeat_vars = []
    for q in texts:
        q_ids = tokenizer.encode(q, max_len=config.MAX_LEN - 1)
        src = torch.tensor(
            [[config.SPECIAL_TOKENS["<BOS>"]] + q_ids], dtype=torch.long, device=device
        )
        ws = []
        for _ in range(repeats):
            # tgt_in 用 src 截断占位（不超过模型 max_len），仅取 router 权重
            tgt_in = src[:, : min(model.max_len, src.size(1))]
            out = model(src, tgt_in, stage=config.STAGE)
            w = out["router_weights"][0].cpu()          # [4]
            ws.append(w)
            if weights_sum is None:
                weights_sum = w.clone()
            else:
                weights_sum += w
            count += 1
        ws_mat = torch.stack(ws)
        repeat_vars.append(ws_mat.var(dim=0).mean().item())

    avg_w = weights_sum / max(count, 1)
    entropy = -(avg_w * torch.log(avg_w.clamp_min(1e-9))).sum().item()
    return {
        "marginal_router_entropy": entropy,          # log4≈1.386 为上限；<0.7 视为趋向坍缩
        "avg_load": {f"m{i}": round(float(avg_w[i]), 3) for i in range(4)},
        "repeat_weight_var": sum(repeat_vars) / max(len(repeat_vars), 1),
        "n_samples": count,
    }


def print_generation_report(metrics):
    print("\n[评测] 生成质量基线")
    print(f"  字符级 F1:      {metrics['f1_mean']:.3f}")
    print(f"  命中率(≥0.35):  {metrics['hit_rate@0.35']:.1%}")
    print(f"  平均长度:       {metrics['avg_len']:.1f} 字")
    print(f"  空洞输出率:     {metrics['empty_rate']:.1%}")
    for q, resp, ref in metrics["samples"]:
        print(f"  Q: {q}\n  G: {resp}\n  R: {ref}")


def print_router_report(metrics):
    print("\n[评测] 分区稳定性基线")
    print(f"  边际路由熵:       {metrics['marginal_router_entropy']:.3f} (log4≈1.386 上限, <0.7 视为坍缩)")
    print(f"  四模块平均载荷:   {metrics['avg_load']}")
    print(f"  重复生成权重方差: {metrics['repeat_weight_var']:.4f} (越小越稳定)")
    print(f"  采样样本数:       {metrics['n_samples']}")


# ==================== 训练（原 anwan/train.py） ====================

class UnifiedTrainer:
    def __init__(self, model, tokenizer, device="cpu", checkpoint_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_path = checkpoint_path or config.CHECKPOINT_FILE
        self.optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
        # scheduler 在 train_stage 内按阶段重建：每段独立余弦退火（T_max=该段 epoch 数），
        # 避免跨阶段共用一个 30 步周期导致 SFT 阶段从 0 回升的错误学习率曲线
        self.scheduler = None
        self.best_loss = float("inf")
        self.history = {"lm": [], "sft": []}
        # ---- EMA 影子权重与最优权重快照（P0-2 升级）----
        self.ema_state = None            # 指数移动平均影子权重（训练平滑/推理回退）
        self.ema_decay = 0.999           # EMA 衰减系数
        self.best_state = None           # 验证集最优权重快照（早停后恢复用）
        self.best_val = float("inf")     # 历史最优验证损失
        # ---- AMP 自动混合精度（Orin Nano 8G 内存优化）----
        self.amp_enabled = getattr(config, "AMP_ENABLED", True) and torch.cuda.is_available()
        # torch.cuda.amp.* 在 torch>=2.4 已废弃，统一走 torch.amp 命名空间
        self._amp_dtype = getattr(torch, config.AMP_DTYPE, torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled) if self.amp_enabled else None

    # ---------------- 训练一个 epoch ----------------
    def _run_epoch(self, batches, stage, train=True, with_intent=True):
        self.model.train(train)
        total_loss = 0.0
        seq_loss_sum = 0.0
        n_batches = 0
        # ---- 跨轮情感状态：训练时必须真的串起来 ----
        # 原实现每个 batch 都传 emotion_state=None，LSTM-B 永远从零状态起步，
        # 而推理时 InferenceEngine 每轮都传入非零状态 —— 训练/推理输入分布
        # 不一致，"有状态情感通道"这个能力从未被训练过。改为在 epoch 内把
        # new_state 回喂给下一个 batch，让模型真正学会接收/演化情感状态。
        emo_state = None
        prev_emotion_vec = None   # 上一批显式情感向量（跨轮监督 + 生成注入）
        emo_resets = 0
        emo_carried_in = 0        # 真正"收到"了上一批状态的 batch 数
        h_norms = []
        emo_dir_sum = 0.0
        emo_vec_sum = 0.0
        moe_aux_sum = 0.0         # MoE 负载均衡 loss 累加

        for batch in batches:
            if train:
                self.optimizer.zero_grad()
            # 状态批维度必须与本 batch 一致，否则 h[:, si] 会静默串行
            # （索引不越界、不报错，但取到的是别的样本的状态）。末批通常不满，
            # 尺寸不匹配时直接重置，宁可少串一串也不能用错状态。
            B_now = batch["src"].size(0)
            if emo_state is not None:
                h_chk = emo_state[0]
                batch_dim = h_chk.size(1) if h_chk.dim() >= 3 else h_chk.size(0)
                if batch_dim != B_now:
                    emo_state = None
                    prev_emotion_vec = None
                    emo_resets += 1
            if prev_emotion_vec is not None and prev_emotion_vec.size(0) != B_now:
                prev_emotion_vec = None        # 向量也跨 batch，尺寸不匹配同样重置
            # 数值护栏：LSTM 状态在长链上可能发散，炸了就重置而不是毒化整个 epoch
            if emo_state is not None:
                hn = float(emo_state[0].norm())
                if not (hn == hn) or hn > config.EMOTION_STATE_MAX_NORM:
                    emo_state = None
                    emo_resets += 1
                else:
                    emo_carried_in += 1
                    h_norms.append(hn)

            # AMP 自动混合精度前向（Orin Nano 省 ~40% 显存）
            with torch.amp.autocast("cuda", enabled=self.amp_enabled,
                                    dtype=self._amp_dtype):
                out = self.model(
                    batch["src"],
                    batch["tgt_in"],
                    emotion_state=emo_state,
                    stage=stage,
                    return_intent=with_intent and batch["intents"] is not None,
                    intent_labels=batch["intents"] if (with_intent and batch["intents"] is not None) else None,
                    prev_emotion_vec=prev_emotion_vec,
                )
                losses = self.model.compute_loss(
                    out,
                    batch["tgt_out"],
                    intent_labels=batch["intents"] if (with_intent and batch["intents"] is not None) else None,
                    intent_loss_weight=config.INTENT_LOSS_WEIGHT,
                    prev_emotion_vec=prev_emotion_vec,
                )
            if train:
                # AMP 梯度缩放反向传播（避免 float16 下溢）
                if self.scaler is not None:
                    self.scaler.scale(losses["total"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
                    self.optimizer.step()
                self._update_ema()
            # 回喂情感状态（训练时 detach 跨 batch 传递；验证时重置防泄漏）
            if train:
                new_state = out.get("new_state")
                if new_state is not None:
                    emo_state = tuple(t.detach().clone() for t in new_state)
                nv = out.get("emotion_vec")
                if nv is not None:
                    prev_emotion_vec = nv.detach().clone()
            else:
                emo_state = None
                prev_emotion_vec = None

            total_loss += float(losses["total"].detach())
            seq_loss_sum += float(losses["seq_loss"])
            emo_dir_sum += float(losses.get("emotion_dir", 0.0))
            emo_vec_sum += float(losses.get("emotion_vec", 0.0))
            moe_aux_sum += float(losses.get("moe_aux", 0.0))
            n_batches += 1

        n = max(n_batches, 1)
        g = getattr(self.model, "emotion_gate", None)
        gate = float(g.detach()) if hasattr(g, "detach") else float(g or 1.0)
        return {"total": total_loss / n, "seq": seq_loss_sum / n, "n_batches": n_batches,
                "emo_resets": emo_resets, "emo_carry": emo_carried_in,
                "emo_h_norm": (sum(h_norms) / len(h_norms)) if h_norms else 0.0,
                "emotion_gate": gate, "emo_dir": emo_dir_sum / n,
                "emo_vec": emo_vec_sum / n,
                "moe_aux": moe_aux_sum / n}

    # ---------------- EMA 影子权重 ----------------
    @torch.no_grad()
    def _update_ema(self):
        """指数移动平均：ema = decay*ema + (1-decay)*current；首次以当前权重初始化"""
        if self.ema_state is None:
            self.ema_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            return
        decay = self.ema_decay
        for k, v in self.model.state_dict().items():
            if k in self.ema_state:
                self.ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

    # 渐进显式化支路：零初始化、靠训练一点点长出来。这类权重不能随 best/EMA 快照
    # 整体回滚 —— 早停触发点往往在它们还没长起来的那一轮，一次性回滚会把刚学出来的
    # 情感塑形与半专家直接抹回 0（实测复现过：reload 后 out_projs abs.sum = 0）。
    _GRADUAL_KEYS = ("lstm_b_semi.out_projs", "emo_semi_gate", "emotion_vec_head", "vec_proj")

    def apply_best_or_ema(self):
        """训练结束后恢复最优权重：优先 best_state（验证最优），否则退回 EMA 影子权重。
        渐进显式化支路例外 —— 保留当前（最新）值，只回滚主干权重。
        """
        restored = None
        snapshot = None
        if self.best_state is not None:
            snapshot, restored = self.best_state, "best"
        elif self.ema_state is not None:
            snapshot, restored = self.ema_state, "ema"
        if snapshot is None:
            return None
        live = self.model.state_dict()
        keep = {k: v.clone() for k, v in live.items()
                if k in snapshot and any(g in k for g in self._GRADUAL_KEYS)}
        self.model.load_state_dict(snapshot)
        if keep:
            merged = dict(snapshot)
            merged.update(keep)
            self.model.load_state_dict(merged)
            print(f"  [渐进支路] 保留最新值不回滚 {len(keep)} 项"
                  f"（半专家投影/情感头/门控），避免早停把它们抹回零")
        return restored

    # ---------------- 分段训练 ----------------
    def train_stage(self, stage, epochs, pipeline, device=None):
        """stage: "lm" | "sft"；返回 (avg_last_loss, n_epochs)"""
        device = device or self.device
        assert stage in ("lm", "sft")
        path_stage = stage
        # 关键修复：上一阶段 checkpoint 恢复的 optimizer lr 可能已退火到 0，
        # 必须在此重置为初始 LR 并重建本阶段独立的余弦退火周期（T_max=该段 epoch 数）
        for pg in self.optimizer.param_groups:
            pg["lr"] = config.LR
            pg["initial_lr"] = config.LR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1))

        if stage == "lm":
            batches = list(pipeline.lm_batches(config.BATCH_SIZE, device))
            with_intent = False
        else:
            batches = list(pipeline.sft_batches(config.BATCH_SIZE, device))
            with_intent = True

        # ---- 训练/验证切分（防静默不训练）----
        # 原实现 n_val = max(int(len*0.15), 1)：当总批次只有 1 个时会把唯一的
        # 批次划进验证集，train_batches 变空 → _run_epoch 返回 0.0，训练一步
        # 都没跑却看着 loss 正常下降，并把未训练的初始权重存成"最优"。
        n_val = int(len(batches) * (1 - config.SFT_TRAIN_RATIO))
        if len(batches) >= 4:
            n_val = max(n_val, 1)
        else:
            n_val = 0                       # 批次太少不切验证集
        val_batches = batches[:n_val]
        train_batches = batches[n_val:] or batches      # 兜底：训练集永不为空
        if n_val == 0:
            print(f"  [!] 批次仅 {len(batches)} 个（<4），跳过验证集切分；"
                  f"本次训练的 loss 不具备泛化参考性，建议扩充语料")
        patience = 0
        best_val = float("inf")
        print(f"\n[stage={path_stage}] 批次: 训练 {len(train_batches)} / 验证 {len(val_batches)}")

        for epoch in range(1, epochs + 1):
            # 竞争强度衰减：早期 soft（temp=1.0, weight=0.05），后期趋近 WTA（temp=0.3, weight=0.5）
            temp = max(0.3, 1.0 - (epoch - 1) / max(epochs, 1) * 0.7)
            comp_weight = min(0.5, 0.05 + (epoch - 1) / max(epochs, 1) * 0.45)
            self.model._competition_temp = temp
            self.model._competition_weight = comp_weight
            # emo_state 在 _run_epoch 内部是局部变量，每个 epoch 独立起链：
            # 推理时是一段连续对话，训练时每轮重置可避免状态链无限增长。
            tr = self._run_epoch(train_batches, config.STAGE, train=True, with_intent=with_intent)
            train_loss = tr["total"]
            if val_batches:
                va = self._run_epoch(val_batches, config.STAGE, train=False, with_intent=with_intent)
                val_loss, val_seq = va["total"], va["seq"]
            else:
                val_loss = val_seq = float("nan")
            self.scheduler.step()
            self.history[path_stage].append({"epoch": epoch, "train": train_loss,
                                             "val": val_loss, "val_seq": val_seq})

            # 早停/最优权重只看验证集的 seq_loss（next-token 交叉熵）：
            # 原实现用 total（含 intent 加权 8× 与 router 正则）选最优，
            # 正则项的抖动会污染"泛化最好"这个判断。
            monitor = val_seq if val_batches else train_loss
            if monitor < best_val:
                best_val = monitor
                patience = 0
                if self.best_state is None or best_val < self.best_val:
                    self.best_val = best_val
                    self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if epoch % 5 == 0 or epoch == epochs or patience >= config.PATIENCE:
                vs = f"{val_seq:.4f}" if val_batches else "n/a"
                # 同时给出 total 与 seq：total 含 intent×8 与 router 正则，
                # 量级远大于纯交叉熵，只看 total 会误判训练是否收敛。
                # 情感诊断：carry/norm/gate 三个数能判断"感情系统"是否真在工作
                #   carry  = 本轮有多少 batch 真的继承了上一 batch 的情感状态
                #   norm   = 状态范数均值（全 0 说明情感通道在衰减，状态没传下去）
                #   gate   = 情感注入强度；趋近 0 意味着模型学会"忽略情感"，
                #            此时情感系统等于被关掉了，必须报警
                gate = tr.get("emotion_gate", 1.0)
                warn = "  ⚠️ 情感门控趋零，情感通道被忽略了！" if abs(gate) < 0.05 else ""
                print(
                    f"  [{path_stage} epoch {epoch}/{epochs}] "
                    f"train={tr['total']:.4f}/seq={tr['seq']:.4f} | "
                    f"val_seq={vs} | lr={self.scheduler.get_last_lr()[0]:.2e}"
                )
                print(
                    f"      情感: 串联 {tr.get('emo_carry',0)}/{tr['n_batches']} 批 | "
                    f"状态范数 {tr.get('emo_h_norm',0.0):.2f} | 门控 {gate:.3f}"
                    f" | 重置 {tr.get('emo_resets',0)} 次 | 方向loss {tr.get('emo_dir',0.0):.4f}"
                    f" | PAD向量 {tr.get('emo_vec',0.0):.4f}{warn}"
                )

            if (epoch % config.SAVE_EVERY == 0) or (epoch == epochs):
                self.save_checkpoint(epoch=epoch, stage=path_stage, val_loss=best_val)

            if patience >= config.PATIENCE:
                print(f"  ⏹️ 早停触发（连续 {config.PATIENCE} 轮无提升）")
                break

        # 恢复最优权重（验证集最优优先，EMA 兜底）后再兜底保存
        restored = self.apply_best_or_ema()
        if restored is not None:
            print(f"  [best] 已恢复最优权重（来源={restored}，best_val={self.best_val:.4f}）")
        self.save_checkpoint(epoch=epochs, stage=path_stage, val_loss=self.best_val)  # 兜底保存
        return best_val, epochs

    def full_train(self, pipeline, device=None):
        """阶段一 LM 预训练 → 阶段二 监督调优（同一实例续接，渐进式通路）"""
        device = device or self.device
        print("=" * 60)
        print("[训练] 阶段一：LM 预训练（next-token，语料全覆盖）")
        self.train_stage("lm", config.LM_EPOCHS, pipeline, device)
        print("=" * 60)
        print("[训练] 阶段二：监督调优（问答对 + 意图多任务，渐进式通路）")
        self.train_stage("sft", config.SFT_EPOCHS, pipeline, device)
        print("=" * 60)
        self.lock_and_report(pipeline, device=device)

    # ---------------- checkpoint 管理 ----------------
    def save_checkpoint(self, epoch, stage, val_loss=None):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        # 体积诊断：800M 参数下 model/ema/best/optimizer 四份合计可达约 12GB，
        # Orin Nano 8G 机器上保存本身就可能失败。打印实际大小 + 提供瘦身开关。
        n_params = sum(p.numel() for p in self.model.parameters())
        est_gb = n_params * 4 * 4 / (1024 ** 3)   # 粗估：4 份 fp32 副本
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "best_loss": self.best_loss,
            "val_loss": val_loss,
            "history": self.history,
            "config_snapshot": {
                "vocab_size": self.model.vocab_size,
                "emotion_dim": self.model.emotion_dim,
                "embed_dim": self.model.embed_dim,
                "max_len": self.model.max_len,
                "STAGE": config.STAGE,
            },
            "locked_mask": self.model._locked_mask.cpu().tolist() if self.model._locked_mask is not None else None,
            "amp_scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        # EMA / 最优权重快照：内存里已有，落盘是可选的（Orin 等小内存设备建议关闭）
        if config.CKPT_SAVE_SHADOWS:
            ckpt["ema_state"] = self.ema_state
            ckpt["best_state"] = self.best_state
        else:
            ckpt["ema_state"] = None
            ckpt["best_state"] = None
        ckpt["best_val"] = self.best_val
        t0 = time.time()
        with torch.no_grad():
            torch.save(ckpt, self.checkpoint_path)
        try:
            size_mb = os.path.getsize(self.checkpoint_path) / (1024 ** 2)
        except OSError:
            size_mb = float("nan")
        print(f"  [save] checkpoint -> {self.checkpoint_path} "
              f"(stage={stage}, epoch={epoch}, {size_mb:.0f}MB, {time.time() - t0:.1f}s)")
        if size_mb > 2048:
            print(f"  [!] checkpoint 超过 2GB（参数 {n_params/1e6:.0f}M，粗估含影子权重约 {est_gb:.1f}GB）。"
                  f"小内存设备请在 config 里设 CKPT_SAVE_SHADOWS = False 再训，"
                  f"否则保存过程可能直接 OOM。")
        if not config.CKPT_SAVE_SHADOWS:
            print(f"  [i] CKPT_SAVE_SHADOWS=False：本次未落盘 ema/best 快照，"
                  f"续训将从当前权重继续（best_val={self.best_val:.4f} 仍会记录）")

    def load_checkpoint(self, device=None, strict=False):
        """加载 checkpoint；返回 (stage, epoch, best_val)。
        仅加载磁盘 data/models/anwan_v2.pt；无磁盘时提示并返回 None（从零训练）。
        strict=False（默认）：缺失键保持当前模型初始化（新增模块权重零初始化），
        以 config_snapshot 核对结构与旧 checkpoint 兼容；升级后旧 ckpt 可加载、可回退。
        维度不匹配时逐参数做形状适配迁移（截断/扩展 + 初始化值填充），
        并**绕过 load_state_dict**（shape mismatch 会抛 RuntimeError），
        改用逐参数 copy_ 赋值，保证不崩溃。"""
        device = device or self.device
        if not os.path.exists(self.checkpoint_path):
            print(f"  [从零] 无 checkpoint：{self.checkpoint_path}（将从零训练）")
            return None, 0, float("inf")
        ckpt = torch.load(self.checkpoint_path, map_location=device)
        ckpt_state = dict(ckpt["model_state"])
        # 结构核对：config_snapshot 与当前模型不匹配时仅提示，仍尽力加载
        snap = ckpt.get("config_snapshot", {})
        if snap:
            issues = []
            for key, cur in (("vocab_size", self.model.vocab_size),
                             ("emotion_dim", self.model.emotion_dim),
                             ("embed_dim", self.model.embed_dim),
                             ("max_len", self.model.max_len)):
                if snap.get(key) is not None and snap[key] != cur:
                    issues.append(f"{key}: ckpt={snap[key]} cur={cur}")
            if issues:
                print(f"  [!] checkpoint 结构差异（{'; '.join(issues)}），将逐参数形状适配迁移")
        # 逐参数形状适配 + 绕过 load_state_dict（shape mismatch 会抛 RuntimeError）
        cur_state = self.model.state_dict()
        adapted_count = 0
        for name, cur_param in cur_state.items():
            if name not in ckpt_state:
                adapted_count += 1
                continue
            ck_param = ckpt_state[name]
            if cur_param.shape != ck_param.shape:
                adapted_count += 1
                # 逐维度最小公共部分复制，剩余保持当前初始化值
                min_dims = [min(a, b) for a, b in zip(cur_param.shape, ck_param.shape)]
                slices_cur = tuple(slice(0, d) for d in min_dims)
                slices_ck = tuple(slice(0, d) for d in min_dims)
                with torch.no_grad():
                    cur_param[slices_cur].copy_(ck_param[slices_ck].to(cur_param.device, cur_param.dtype))
            else:
                with torch.no_grad():
                    cur_param.copy_(ck_param.to(cur_param.device, cur_param.dtype))
        if adapted_count:
            print(f"  [适配] {adapted_count} 个参数做了形状迁移（截断/扩展 + 初始化值填充）")
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.history = ckpt.get("history", {"lm": [], "sft": []})
        # 恢复 EMA / 最优权重快照（加载后权重即最优，快照供后续训练继续维护）
        self.ema_state = ckpt.get("ema_state")
        self.best_state = ckpt.get("best_state")
        self.best_val = ckpt.get("best_val", float("inf"))
        # 严格模式校验：ckpt 有 best_state 但 best_val 为 inf（旧版数据不一致），
        # 则以 best_state 实际 loss 为准，防止训练恢复后误用无效快照
        if self.best_state is not None and self.best_val == float("inf"):
            self.best_val = ckpt.get("val_loss", float("inf"))
        # 恢复分化锁定掩码（否则重启后硬门控失效，等于没锁）
        locked = ckpt.get("locked_mask")
        if locked is not None:
            try:
                self.model._locked_mask = torch.tensor(locked, dtype=torch.float32, device=device)
                print(f"  [OK] 分化锁定掩码已恢复: {[int(v) for v in locked]}")
            except Exception as e:
                print(f"  [warn] locked_mask 恢复失败（忽略）: {e}")
        # AMP：恢复 GradScaler 状态（旧 checkpoint 无该键则保持默认）
        scaler_state = ckpt.get("amp_scaler")
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as e:
                print(f"  [warn] GradScaler 状态恢复失败（忽略，重新自适应）: {e}")
        print(f"  [OK] checkpoint 加载：stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')}（来自 {self.checkpoint_path}）")
        return ckpt.get("stage"), ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))

    # ---------------- 训练后评测 ----------------
    @torch.no_grad()
    def evaluate(self, pipeline, device=None, n=20):
        device = device or self.device
        gen = evaluate_generation(self.model, self.tokenizer, pipeline, n=n, device=device)
        router = evaluate_router(self.model, self.tokenizer, n=n, device=device)
        print_generation_report(gen)
        print_router_report(router)
        return {"generation": gen, "router": router}

    # ---------------- 分化锁定与报告（训练后自动固化功能区）----------------
    @torch.no_grad()
    def lock_and_report(self, pipeline, device=None):
        """训练后自动执行分化锁定并输出模块特化报告"""
        device = device or self.device
        # 构建 texts_by_intent：取 pipeline 中前 50 个 QA 对
        texts_by_intent = {}
        labeler = UnifiedIntentLabeler()
        for q, _ in pipeline.qa_pairs[:50]:
            intent_name, score = labeler.label(q)
            texts_by_intent.setdefault(intent_name, []).append(q)
        report = self.model.lock_specialization(self.tokenizer, texts_by_intent, device=device)
        print("\n" + "=" * 56)
        print("[模块特化报告] 大脑分区固化结果")
        print("=" * 56)
        if report["global"]["locked"]:
            print(f"  锁定状态: 已固化（硬门控生效）")
            print(f"  边际路由熵: {report['global']['entropy']:.3f}（<0.7 视为充分分化）")
            print(f"  全局载荷: {report['global']['avg_load']}")
            print(f"  锁定掩码: {report['global']['mask']}（1=允许激活 0=抑制）")
        else:
            print("  锁定状态: 未固化（数据不足）")
        print("  意图-模块映射:")
        for intent, info in report["intents"].items():
            mod_name = ["内容TF", "情感LSTM-B", "检索TF-1", "生成TF-2"][info["top_module"]]
            print(f"    {intent:12s} -> {mod_name} (权重={info['top_weight']:.3f})")
        print("=" * 56)
        return report


# ==================== 推理引擎（原 anwan/inference.py） ====================

# 模块名（router 输出 4 维权重对应说明）
MODULE_NAMES = ["内容TF", "情感LSTM-B", "检索TF-1", "生成TF-2"]

# 慢轨相似度阈值：低于此不采用（保质量底线）
SIM_THRESHOLD = 0.62
# 否定语境抑制：这些问题意味着用户情绪倾向，不应被字面相似拖进"调侃"语料
NEGATIVE_CONTEXT = {
    "讨厌", "烦", "生气", "难过", "伤心", "哭", "不理", "滚",
    "笨", "没用", "吵", "安静", "闭嘴", "删", "关掉", "走开", "别来",
}


def _clean_text(text: str) -> str:
    """全角转半角 + 去除首尾空白 + 压缩内部空白"""
    if not text:
        return ""
    text = text.strip().replace("\u3000", " ").replace("\u200b", "")
    # 全角转半角
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    text = "".join(out)
    return re.sub(r"\s+", " ", text).strip()


class InferenceEngine:
    """安晚对话推理引擎（快轨 / 慢轨 / 模型生成 / 兜底 四层）"""

    def __init__(self, model, tokenizer, device="cpu", stage="main",
                 qa_pairs=None, temperature=0.8, top_p=0.9, max_len=25):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stage = stage
        self.temperature = temperature
        self.top_p = top_p
        self.max_len = max_len

        # 语料索引（快轨 dict：问句 → 答案列表；慢轨样本：问句/答案全文）
        self._fast_index = {}      # 归一化问句 -> 答案列表
        self._slow_samples = []    # [(归一化问句, 答案列表)]
        if qa_pairs:
            self.build_index(qa_pairs)

        # 跨轮情感状态（LSTM-B 的 (h, c)）；None 表示零初始化
        self.emotion_state = None
        # 显式情感向量（PAD 三维：愉悦/唤醒/支配）；跨轮持久化 + 生成注入 + 可解释输出
        self.emotion_vec = None
        # P1-3：情感状态持久化文件（重启恢复 + 时间衰减）
        self.emotion_file = config.EMOTION_STATE_FILE

        # 显式多轮历史（P1-1）：[(user_q, bot_reply), ...]，最近 HISTORY_TURNS 轮
        self.history = []

        # 外部长期记忆（P1-2）：由 cmd_chat 挂载 ExternalMemory 实例
        self.memory = None
        # 生词本：生僻词/外语查询。模型答不了这类，必须走词典而不是自由生成
        self.wordbook = None

        self.rounds = 0
        self.hit_log = []          # [(问句, 命中层, 是否模型生成)]

    # ---------------------------------------------------------- 语料索引 --
    def build_index(self, qa_pairs):
        """从 (问句, 答案) 列表构建快/慢轨索引；可重复调用追加"""
        for q, answers in qa_pairs:
            qc = _clean_text(q)
            if not qc:
                continue
            if isinstance(answers, str):
                answers = [answers]
            answers = [_clean_text(a) for a in answers if a and _clean_text(a)]
            if not answers:
                continue
            if qc not in self._fast_index:
                self._fast_index[qc] = []
                self._slow_samples.append((qc, self._fast_index[qc]))
            self._fast_index[qc].extend(
                a for a in answers if a not in self._fast_index[qc]
            )

    def update_corpus(self, qa_pairs):
        """训练后追加新语料（增量索引）"""
        self.build_index(qa_pairs)
        return len(self._fast_index)

    def rebuild_index(self, qa_pairs):
        """重新匹配：清空并重建快/慢轨索引（不训练，仅 rebuild 语料索引）"""
        self._fast_index = {}
        self._slow_samples = []
        self.build_index(qa_pairs)
        return len(self._fast_index)

    # ---------------------------------------------------------- 快 / 慢轨 --
    def _fast_match(self, question: str):
        """快轨：精确 / 包含命中。返回 (答案, 提示词)，未命中返回 None"""
        q = question
        # 1) 精确命中
        if q in self._fast_index:
            return self._fast_index[q], "exact"
        # 2) 双向包含（问句是语料问题子串 / 语料问题是问句子串，且长度差可控）
        for k, answers in self._fast_index.items():
            if q and (k in q or q in k):
                if len(k) >= 1 and abs(len(k) - len(q)) <= max(3, len(q) // 2):
                    return answers, "contains"
        return None

    def _slow_match(self, question: str):
        """慢轨：difflib 相似度 + 否定语境抑制。返回 (答案, 相似度) 或 None"""
        best_q, best_ratio, best_answers = None, 0.0, None
        for k, answers in self._slow_samples:
            if not k:
                continue
            ratio = difflib.SequenceMatcher(None, question, k).ratio()
            if ratio > best_ratio:
                best_ratio, best_q, best_answers = ratio, k, answers
        if best_ratio >= SIM_THRESHOLD and best_answers:
            # 否定语境抑制：用户问题含负面情绪词时，不采用相似语料里
            # 含"调侃/针锋相对"倾向的回复，避免答非所愿
            if any(w in question for w in NEGATIVE_CONTEXT):
                if any(w in best_q for w in NEGATIVE_CONTEXT):
                    pass  # 双方负面语境匹配，允许采用（如"不理我"类语料）
                else:
                    return None
            return best_answers, round(best_ratio, 3)
        return None

    @torch.no_grad()
    def _model_reply(self, question: str, intent: str, src=None):
        """模型自回归生成回复，返回 (文本, new_state, router 权重, 熵)。
        router 权重由 generate 第 1 步直接返回（等效原 _router_probe，免二次前向）。
        - src 为 None 时退化为单轮编码；否则使用调用方拼好的 src（可含多轮历史/记忆注入）"""
        # 与训练侧 sft_batches 一致：BOS 在位置 0
        if src is None:
            q_ids = self.tokenizer.encode(question, max_len=self.model.max_len - 1)
            seq = [BOS_IDX] + list(q_ids)
            seq = seq[: self.model.max_len]
            src = torch.tensor([seq], dtype=torch.long, device=self.device)
        elif not isinstance(src, torch.Tensor):
            src = torch.tensor([list(src)[: self.model.max_len]],
                               dtype=torch.long, device=self.device)

        tokens, new_state, w, ent, emotion_vec = self.model.generate(
            src,
            emotion_state=self.emotion_state,
            max_len=self.max_len,
            temperature=self.temperature,
            top_p=self.top_p,
            strategy="nucleus",
            stage=self.stage,
            prev_emotion_vec=self.emotion_vec,
        )
        # 去掉前缀 BOS 与 EOS
        ids_out = [int(i) for i in tokens[0].tolist()][1:]
        if ids_out and ids_out[-1] == EOS_IDX:
            ids_out = ids_out[:-1]
        text = self.tokenizer.decode(ids_out)
        text = _clean_text(text)

        # ---- AWIL：输出本次推理的模块通信流水账 ----
        if getattr(self.model, "awil_bus", None) is not None and self.model.awil_bus.enabled and len(self.model.awil_bus) > 0:
            print("\n[AWIL 模块通信流水账]")
            print(self.model.awil_bus.dump(last_n=20))
            print("[/AWIL]\n")
            self.model.awil_bus.clear()

        return text, new_state, w, ent, emotion_vec

    # ---------------------------------------------------------- 质量门 --
    @staticmethod
    def _quality_gate(text: str) -> bool:
        """模型输出质量底线：非空、有实际词汇、非单一字符重复刷屏、中文占比达标"""
        if not text or len(text) < config.REPLY_MIN_LEN:
            return False
        # 空洞输出：全标点 / 全重复（如 "~~~~~~~"、"哈哈哈" * 20）
        uniq = set(text)
        if len(uniq) <= 2 and len(text) >= 3:
            return False
        # 单个字符连续刷屏 > 12 次
        m = re.search(r"(.)\1{11,}", text)
        if m:
            return False
        letters = sum(1 for ch in text if ch.isalpha() or ("\u4e00" <= ch <= "\u9fff"))
        if len(text) >= 3 and letters / len(text) < 0.4:
            return False
        return True

    # ---------------------------------------------------------- 兜底词库 --
    @staticmethod
    def _fallback(intent: str) -> str:
        return {
            "greeting": "你好呀～",
            "farewell": "拜拜～我会想你的喵！",
            "identity": "我是安晚呀，你的专属小猫娘～",
            "state_query": "本机状态良好！精力充沛喵！(๑•̀ㅂ•́)و✧",
            "emotion_share": "抱抱你，别难过，我一直在你身边喵～",
            "affection": "我也最喜欢你啦～（蹭蹭）",
            "teasing": "哼！我记住你了喵！",
            "question": "唔……这个问题我还在学习，下次再告诉你喵～",
            "admin_cmd": "收到，已记录喵～",
            "egg_trigger": "彩蛋？被我藏起来啦，再找找看～",
            "action_request": "好呀～这就来！",
            "chat": "喵～我在听你说呢～",
            "lang_learning": "你想学哪句？我可以教你怎么说～不过新词我得先学一下喵。",
            "rare_word": "这个词我还没学过喵～你教我一次我就记住啦！",
        }.get(intent, "喵～我在听你说呢～")

    # ---------------------------------------------------------- 对外接口 --
    def reset_emotion(self):
        """温和重置：清空跨轮情感状态、多轮历史与命中日志（保留外部记忆与已学权重）。
        同时清除情感持久化文件，避免下次启动恢复旧情感。"""
        self.emotion_state = None
        self.emotion_vec = None
        self.history = []
        self.hit_log = []
        self.rounds = 0
        try:
            if self.emotion_file and os.path.isfile(self.emotion_file):
                os.remove(self.emotion_file)
        except OSError:
            pass

    # ---------------------------------------------------------- 情感持久化 --
    def load_persisted_emotion(self):
        """重启恢复情感状态（含时间衰减）；已有机内状态时不动"""
        if self.emotion_state is None:
            st = load_emotion_state(self.device, self.emotion_file)
            if st is not None:
                self.emotion_state = (st[0], st[1])
                vec = st[2] if len(st) > 2 else None
                if vec is not None:
                    self.emotion_vec = vec
                print("[情感] 已恢复上次跨轮情感状态与 PAD 向量（含离线时间衰减）")
        return self.emotion_state

    def save_persisted_emotion(self):
        """情感状态落盘（respond 每轮有效生成后调用）"""
        save_emotion_state(self.emotion_state, self.emotion_file,
                           emotion_vec=self.emotion_vec)

    def emotion_summary(self):
        """情感通道摘要：PAD 显式情感向量（愉悦/唤醒/支配）+ L2 范数 / 标准差"""
        h = self.emotion_state[0] if self.emotion_state is not None else None
        base = {"active": self.emotion_state is not None,
                "h_norm": round(float(h.norm(2).item()), 4) if h is not None else 0.0,
                "h_std": round(float(h.std().item()), 4) if h is not None else 0.0}
        if self.emotion_vec is None:
            base["emotion_vec"] = None
            base["emotion_label"] = None
            return base
        v = [round(float(x), 3) for x in self.emotion_vec.detach().cpu().reshape(-1).tolist()]
        base["emotion_vec"] = v
        base["emotion_label"] = emotional_label(v)
        return base

    def respond(self, question: str, scope=None, profile=None):
        """
        单轮推理。返回 (reply_text, meta)。
        meta: {"layer", "intent", "router_weights", "router_entropy",
               "sim_ratio", "emotion", "profile", "scope"}
        优先级：外部记忆 → 模型生成 → 慢轨相似 → 快轨精确/包含兜底 → fallback
        - scope: 发送者/会话 id，记忆检索的隔离边界（弹幕场景必须传，否则会
                 把别人的记忆注入当前回复）
        - profile: "barrage" | "dm"，检索档位
        """
        q = _clean_text(question)
        if not q:
            return "", {"layer": "fallback", "intent": "chat",
                        "router_weights": None, "router_entropy": None,
                        "sim_ratio": None, "emotion": self.emotion_summary()}

        prof = ExternalMemory.get_profile(profile)
        labeled = UnifiedIntentLabeler.label(q)
        # label 返回 (意图名, 置信度)；内部使用字符串意图名
        intent = labeled[0] if isinstance(labeled, tuple) else labeled
        meta = {"intent": labeled, "router_weights": None,
                "router_entropy": None, "sim_ratio": None,
                "emotion": None, "memory_hits": 0, "history_turns": 0,
                "profile": profile or config.DEFAULT_PROFILE, "scope": scope}

        # 1) 外部记忆检索（最优先）：连续记忆层
        memory_hits = []
        conflict_info = {"detected": False, "groups": 0, "dropped": 0}
        if self.memory is not None:
            scored = self.memory.search(q, scope=scope, profile=profile)
            floor = prof["min_score"]
            memory_hits = [t for _s, t, _mid in scored if _s >= floor]
            # 冲突检测与仲裁：多条记忆语义矛盾时按优先级保留最优
            if len(memory_hits) >= 2:
                conflict_groups = self.memory.conflict_detect(scored)
                if conflict_groups:
                    keep_ids, drop_ids = self.memory.conflict_arbitrate(conflict_groups)
                    conflict_info = {
                        "detected": True,
                        "groups": len(conflict_groups),
                        "dropped": len(drop_ids),
                    }
                    # 只保留仲裁胜出的记忆
                    keep_set = set(keep_ids)
                    memory_hits = [t for _s, t, _mid in scored
                                   if _mid in keep_set and _s >= floor]
        if memory_hits:
            # 记忆命中时，把记忆内容作为已知事实注入上下文，让模型基于记忆生成连贯回复
            hist = list(self.history)
            mem_src = build_src_with_memory(self.tokenizer, memory_hits, hist, q,
                                            max_len=self.model.max_len,
                                            mem_max_tokens=prof["max_tokens"],
                                            top_k=prof["top_k"])
            text, new_state, w, ent, new_vec = self._model_reply(q, intent, src=mem_src)
            meta["memory_hits"] = len(memory_hits)
            meta["conflict"] = conflict_info
            if self._quality_gate(text):
                self.emotion_state = new_state
                self.emotion_vec = new_vec
                try:
                    self.save_persisted_emotion()
                except Exception:
                    pass
                self.history.append((q, text))
                if len(self.history) > config.HISTORY_TURNS:
                    self.history = self.history[-config.HISTORY_TURNS:]
                self.rounds += 1
                meta["layer"] = "memory"
                meta["router_weights"] = {MODULE_NAMES[i]: round(v, 3)
                                          for i, v in enumerate(w)}
                meta["router_entropy"] = round(ent, 3)
                meta["history_turns"] = len(self.history)
                self._log(q, meta["layer"], True)
                return text, self._finalize_meta(meta)

        # 1.5) 生词本：生僻词/外语查询。放在模型生成之前 ——
        # 让从零训练的模型去自由生成"龘"的意思 = 编一个自信的错答案。
        # 查到就直接答；查不到也要给可执行的引导（怎么往词本里加）。
        # 路由条件：意图判定为查词，**或**该词已经在生词本里（用户教过）。
        # 后者很重要：单个拉丁词（hello / serendipity）歧义太大，不能靠意图
        # 判定劫持成查词；但只要用户教过我，就该无条件答出来。
        if self.wordbook is not None:
            term = extract_lookup_term(q)
            wants = intent in ("rare_word", "lang_learning")
            known = bool(term) and self.wordbook.get(term) is not None
            if term and (wants or known):
                entry = self.wordbook.render(term) if known else None
                if entry:
                    meta["layer"] = "wordbook"
                    meta["lookup_term"] = term
                    self.history.append((q, entry))
                    if len(self.history) > config.HISTORY_TURNS:
                        self.history = self.history[-config.HISTORY_TURNS:]
                    self.rounds += 1
                    self._log(q, meta["layer"], False)
                    return entry, self._finalize_meta(meta)
                # 查不到：给"我不知道 + 怎么加"的诚实回复，不进模型生成
                hint = (f"「{term}」还不在我的生词本里喵，我不敢乱讲～\n"
                        f"你可以直接教我：对话里输入 /生词 加 {term} <释义>，"
                        f"下次我就能答啦。")
                meta["layer"] = "wordbook_miss"
                meta["lookup_term"] = term
                self._log(q, meta["layer"], False)
                return hint, self._finalize_meta(meta)

        # 2) 模型生成（无相关记忆时）
        hist = list(self.history)
        if hist:
            hist_src = build_src_with_memory(self.tokenizer, [], hist, q,
                                             max_len=self.model.max_len)
            text, new_state, w, ent, new_vec = self._model_reply(q, intent, src=hist_src)
        else:
            text, new_state, w, ent, new_vec = self._model_reply(q, intent)
        if self._quality_gate(text):
            self.emotion_state = new_state
            self.emotion_vec = new_vec
            try:
                self.save_persisted_emotion()
            except Exception:
                pass
            self.history.append((q, text))
            if len(self.history) > config.HISTORY_TURNS:
                self.history = self.history[-config.HISTORY_TURNS:]
            self.rounds += 1
            meta["layer"] = "model"
            meta["router_weights"] = {MODULE_NAMES[i]: round(v, 3)
                                      for i, v in enumerate(w)}
            meta["router_entropy"] = round(ent, 3)
            meta["history_turns"] = len(self.history)
            self._log(q, meta["layer"], True)
            return text, self._finalize_meta(meta)

        # 3) 慢轨：相似度匹配（带否定语境抑制）
        slow = self._slow_match(q)
        if slow is not None:
            answers, ratio = slow
            reply = answers[0]
            meta["layer"] = "slow"
            meta["sim_ratio"] = ratio
            self._log(q, meta["layer"], False)
            return reply, self._finalize_meta(meta)

        # 4) 快轨降级为兜底：精确/包含命中
        hit = self._fast_match(q)
        if hit is not None:
            answers, how = hit
            reply = answers[0]
            meta["layer"] = "fast/" + how
            self._log(q, meta["layer"], False)
            return reply, self._finalize_meta(meta)

        # 5) 最终兜底
        meta["layer"] = "fallback"
        self._log(q, meta["layer"], False)
        return self._fallback(intent), self._finalize_meta(meta)

    # ---------------------------------------------------------- 内部 --
    def _finalize_meta(self, meta):
        meta["emotion"] = self.emotion_summary()
        return meta

    def _log(self, question, layer, is_model):
        self.hit_log.append((question, layer, is_model))
        if len(self.hit_log) > 200:
            self.hit_log = self.hit_log[-200:]

# ==================== 情感状态持久化（P1-3） ====================

def emotional_label(vec):
    """PAD 三维 → 可读情绪标签（唤醒为轴、愉悦定正负的粗粒度映射）"""
    try:
        p, a, d = (float(x) for x in vec)
    except (TypeError, ValueError):
        return "平静"
    if a >= 0.55:
        return "兴奋" if p >= 0.2 else ("紧张" if p <= -0.2 else "激动")
    if a <= 0.25:
        return "愉悦安详" if p >= 0.4 else ("低落" if p <= -0.2 else "平静")
    return "开心" if p >= 0.4 else ("不开心" if p <= -0.3 else "平稳")


def save_emotion_state(state, path=None, emotion_vec=None):
    """跨轮情感 (h, c, emotion_vec) 落盘；附带时间戳供离线衰减。state 为 None 时跳过。
    emotion_vec: PAD 显式情感向量（可选；旧调用方不传也不报错）"""
    if state is None:
        return
    try:
        path = path or config.EMOTION_STATE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h, c = state
        data = {"h": h.detach().cpu(), "c": c.detach().cpu(),
                "ts": time.time()}
        if emotion_vec is not None:
            data["emotion_vec"] = emotion_vec.detach().cpu()
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [!] 情感状态保存失败: {e}")


def load_emotion_state(device="cpu", path=None):
    """恢复情感状态；按离线时长做时间衰减（半衰期 EMOTION_DECAY_DAYS 天），
    离线越久情感越平缓（指数衰减而非一刀切归零）。
    返回 (h, c, emotion_vec|None)：兼容旧格式文件（缺失 emotion_vec 键）。"""
    path = path or config.EMOTION_STATE_FILE
    if not os.path.isfile(path):
        return None
    try:
        data = torch.load(path, map_location=device)
        h = data["h"].to(device)
        c = data["c"].to(device)
        ts = float(data.get("ts", time.time()))
        days = max(0.0, (time.time() - ts) / 86400.0)
        vec = data.get("emotion_vec")
        if vec is not None:
            vec = vec.to(device)
        if days > 0:
            decay = math.exp(-days / config.EMOTION_DECAY_DAYS)
            h = h * decay
            c = c * decay
            if vec is not None:
                vec = vec * decay
        return h, c, vec
    except Exception as e:
        print(f"  [!] 情感状态加载失败: {e}")
        return None


# ==================== boot / CLI（精简自 anwan/main.py） ====================


def _ensure_tokenizer():
    """加载词表：磁盘 data/models/tokenizer.json 优先，无磁盘时从零训练最小保底词表"""
    tok = BPEAnwanTokenizer(vocab_size=config.VOCAB_SIZE)
    if os.path.exists(config.TOKENIZER_FILE):
        if tok.load_vocab(config.TOKENIZER_FILE):
            print(f"[词表] 已加载磁盘: {config.TOKENIZER_FILE} (词表 {tok.next_idx}/{config.VOCAB_SIZE})")
            return tok
    # 无磁盘词表：用最小保底语料训练一个基础词表（供从零启动）
    _seed_corpus = [
        ("你好", ["你好喵~"]),
        ("在吗", ["在呢在呢喵~"]),
        ("再见", ["再见喵~"]),
        ("你是谁", ["我是安晚喵~"]),
        ("你喜欢什么", ["安晚喜欢主人喵~"]),
    ]
    texts = []
    for q, replies in _seed_corpus:
        texts.append(q)
        texts.extend(replies)
    tok.train(texts)
    print(f"[词表] 从零训练保底词表: {tok.next_idx}/{config.VOCAB_SIZE} 个 token")
    return tok


# 书写系统 → 语言标签（粗粒度但够用：拉丁字母无法区分英/法/德，归为 latin）
_SCRIPT_RANGES = [
    ("zh",  0x4E00, 0x9FFF), ("zh", 0x3400, 0x4DBF),
    ("ja",  0x3040, 0x30FF), ("ko", 0xAC00, 0xD7AF),
    ("ru",  0x0400, 0x04FF), ("ar", 0x0600, 0x06FF),
    ("he",  0x0590, 0x05FF), ("hi", 0x0900, 0x097F),
    ("th",  0x0E00, 0x0E7F), ("ta", 0x0B80, 0x0BFF),
    ("te",  0x0C00, 0x0C7F), ("el", 0x0370, 0x03FF),
    ("latin", 0x0041, 0x024F),
]


def detect_script(text):
    """取文本的主导书写系统标签"""
    tally = {}
    for ch in text:
        o = ord(ch)
        if o < 0x80:
            if ch.isalpha():
                tally["latin"] = tally.get("latin", 0) + 1
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= o <= hi:
                tally[name] = tally.get(name, 0) + 1
                break
    if not tally:
        return "other"
    # 日语判别：假名是日语独有的标记。纯多数投票会失败 —— "吾輩は猫" 里
    # 汉字 3 个、假名 1 个，会被误判成中文。有假名就判 ja。
    if tally.get("ja", 0) > 0:
        return "ja"
    return max(tally.items(), key=lambda kv: kv[1])[0]


def analyze_langpack(pack_dir, tokenizer=None):
    """语言包体检：逐语种统计文本量 / 字符集 / 词表覆盖率。
    4 万词表分给几十种语言时，**必然有语种被饿死**；这个报告就是用来
    告诉你死的是谁 —— 否则你只能看到"整体 UNK 率还行"，却不知道
    阿拉伯语用户一句都说不通。"""
    stats = {}
    if not pack_dir or not os.path.isdir(pack_dir):
        return stats
    exts = (".txt", ".json", ".md", ".csv", ".tsv")
    files = []
    for root, _d, fs in os.walk(pack_dir):
        for fn in sorted(fs):
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue
        # 相对语言包子目录的层级作为语种名（pack/zh/xxx.txt → zh）
        rel = os.path.relpath(fp, pack_dir)
        parts = rel.split(os.sep)
        lang = parts[0] if len(parts) > 1 else detect_script(content[:4000])
        d = stats.setdefault(lang, {"files": 0, "chars": 0, "texts": 0, "charset": set()})
        d["files"] += 1
        d["chars"] += len(content)
        d["texts"] += 1
        d["charset"].update(ch for ch in content if not ch.isspace())
    out = {}
    for lang, d in stats.items():
        cs = d["charset"]
        rec = {"files": d["files"], "mb": round(d["chars"] / 1e6, 2), "chars": len(cs)}
        if tokenizer is not None:
            covered = sum(1 for ch in cs if ch in tokenizer.word2idx)
            rec["coverage"] = round(covered / max(len(cs), 1) * 100, 1)
            # 方案A：该语种实际分到的 token 预算（软分区视图，排查"语种饥饿"）
            rec["tokens"] = len(tokenizer.vocab_by_script(lang))
        out[lang] = rec
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mb"]))


def print_langpack_report(stats, tokenizer=None):
    if not stats:
        return
    print("  语言包分语种覆盖：")
    print(f"    {'语种':<10} {'文件':>4} {'体积MB':>7} {'字符数':>7} " +
          (f"{'词表覆盖':>8}{'token数':>9}" if tokenizer is not None else ""))
    weak = []
    for lang, r in stats.items():
        line = f"    {lang:<10} {r['files']:>4} {r['mb']:>7.2f} {r['chars']:>7} "
        if tokenizer is not None:
            line += f"{r.get('coverage', 0):>7.1f}%"
            line += f"{r.get('tokens', 0):>9}"
            if r.get("coverage", 100) < 60:
                weak.append(lang)
        print(line)
    if weak:
        print(f"    [!] 覆盖不足 60% 的语种：{', '.join(weak)}")
        print(f"        4 万词表分给多种语言时必然有语种被饿死。要救这些语种，"
              f"要么给它们单独的词表（按语种分文件建），要么把它们从语言包里去掉、"
              f"避免拉低整体质量。")


def collect_corpus_texts(extra_folders=None):
    """扫描所有语料目录，收集用于训练 BPE 词表的纯文本。
    覆盖：train_data/（外部语料）、训练数据/（learn 学习成果）。
    返回 (texts, 来源统计 dict)。"""
    # 注意 LEARN_FOLDER 只存在于模块全局，不在 config namespace 里
    cands = [config.TRAIN_FOLDER, LEARN_FOLDER]
    cands += list(extra_folders or [])
    seen_dir, folders = set(), []
    for f in cands:
        if f and os.path.isdir(f):
            rf = os.path.abspath(f)
            if rf not in seen_dir:
                seen_dir.add(rf)
                folders.append(f)
    texts, info = [], {}
    for folder in folders:
        pairs = load_folder_pairs(folder)      # 该函数自己扫 .txt/.json/.md
        if not pairs:
            continue
        for q, a in pairs:
            texts.append(q)
            texts.extend(a if isinstance(a, list) else [a])
        info[folder] = len(pairs)
    # 去掉完全重复的文本（BPE 词频统计对重复很敏感）
    seen, uniq = set(), []
    for t in texts:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq, info


def build_tokenizer_from_corpus(folder=None, force=False, epochs_probe=200,
                                 langpack=None, vocab_size=None, merge_cap=None):
    """用真实语料训练并落盘 BPE 词表。
    动机：_ensure_tokenizer() 只会用 5 对硬编码种子语料兜底（实测仅 50 个
    token），之后磁盘优先 → 词表永远停在 50，新语料全部落 UNK。
    必须由真实语料重建词表，否则"多喂语料"这件事完全不成立。
    返回 (tokenizer, 统计 dict)。"""
    if vocab_size:
        config.VOCAB_SIZE = vocab_size
    if merge_cap:
        config.BPE_MERGE_CAP = merge_cap
    if os.path.exists(config.TOKENIZER_FILE) and not force:
        tok = BPEAnwanTokenizer(vocab_size=config.VOCAB_SIZE)
        if tok.load_vocab(config.TOKENIZER_FILE):
            print(f"[词表] 已存在，词表 {tok.next_idx}（加 --force 重建）")
            return tok, {"skipped": True, "vocab": tok.next_idx}
    texts, info = collect_corpus_texts([folder] if folder else None)
    if not texts:
        print(f"[!] 在 {config.TRAIN_FOLDER} / {LEARN_FOLDER} 没找到语料")
        print("    请把 .txt/.json/.md 放进 train_data/ 后重试")
        return None, {"error": "no corpus"}
    print(f"[语料] 收集到 {len(texts)} 条去重文本")
    for k, v in info.items():
        print(f"        {k}: {v} 组问答")
    if langpack:
        # 语言包文本也要进 BPE 训练，否则该语种全部落 UNK
        pack_texts = []
        for root, _d, fs in os.walk(langpack):
            for fn in sorted(fs):
                if os.path.splitext(fn)[1].lower() not in (".txt", ".json", ".md", ".csv", ".tsv"):
                    continue
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        pack_texts.append(f.read())
                except OSError:
                    pass
        pack_blob = "\n".join(pack_texts)
        n_before = len(texts)
        texts.extend([ln.strip() for ln in pack_blob.splitlines() if ln.strip()])
        seen, uniq = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        texts = uniq
        print(f"[语言包] {langpack}: {len(pack_blob)/1e6:.1f}MB 文本并入训练语料"
              f"（{n_before} → {len(texts)} 条）")
    est = len(texts) * min(config.BPE_MERGE_CAP, 200) / max(len(texts), 1)
    if len(texts) > 60000:
        print(f"  [!] 语料 {len(texts)} 条较多，BPE 训练是 O(合并数×词表数)，"
              f"可能需要数小时。建议先用 --folder 子集试跑，再全量。")
    t0 = time.time()
    tok = BPEAnwanTokenizer(vocab_size=config.VOCAB_SIZE)
    tok.train(texts)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(config.TOKENIZER_FILE), exist_ok=True)
    tok.save_vocab(config.TOKENIZER_FILE)
    # 用留出样本测 UNK 率：这是判断"词表够不够大"的直接指标
    probe = texts[-epochs_probe:] if len(texts) > epochs_probe else texts
    ids = [i for t in probe for i in _encode_nopad(tok, t)]
    unk = sum(1 for i in ids if i == UNK_IDX)
    rate = unk / max(len(ids), 1)
    if langpack:
        print_langpack_report(analyze_langpack(langpack, tok), tok)
    print(f"[词表] 已保存 {config.TOKENIZER_FILE}")
    print(f"       合并 {len(tok.bpe_merges)}/{config.BPE_MERGE_CAP} | "
          f"词表 {tok.next_idx}/{config.VOCAB_SIZE} | 耗时 {dt:.1f}s")
    print(f"       留出样本 UNK 率 {rate*100:.1f}%")
    if len(tok.bpe_merges) >= config.BPE_MERGE_CAP:
        print(f"  [!] 合并次数已打满 BPE_MERGE_CAP={config.BPE_MERGE_CAP}，"
              f"词表被卡死在 {tok.next_idx}。加语料也不会再涨了 —— "
              f"请调大 config.BPE_MERGE_CAP 后重建。")
    if rate > 0.15:
        print(f"  [!] UNK 率 {rate*100:.0f}% 偏高，建议调大 BPE_MERGE_CAP / VOCAB_SIZE")
    return tok, {"vocab": tok.next_idx, "merges": len(tok.bpe_merges),
                 "unk_rate": rate, "texts": len(texts), "seconds": dt}


def _build_model(tokenizer):
    return UnifiedAnwanModel(
        vocab_size=tokenizer.next_idx,
        embed_dim=config.EMBED_DIM,
        lstm_hidden=config.LSTM_HIDDEN,
        lstm_b_hidden=config.LSTM_B_HIDDEN,
        emotion_dim=config.EMOTION_DIM,
        emotion_vec_dim=config.EMOTION_VEC_DIM,
        num_heads=config.NUM_HEADS,
        trans_layers=config.TRANS_LAYERS,
        decoder_layers=config.DECODER_LAYERS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
        num_intents=len(config.INTENTS),
        tie_weights=config.TIE_WEIGHTS,
        cross_gate_init=config.CROSS_GATE_INIT,
        router_entropy_weight=config.ROUTER_ENTROPY_WEIGHT,
        entropy_floor=config.ROUTER_ENTROPY_FLOOR,
    )


def boot(verbose=True):
    """统一启动：词表 + 模型 + checkpoint（磁盘优先，无磁盘时从零初始化）+ trainer"""
    tok = _ensure_tokenizer()
    model = _build_model(tok)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    trainer = UnifiedTrainer(model, tok, device=device,
                             checkpoint_path=config.CHECKPOINT_FILE)
    stage, epoch, val = trainer.load_checkpoint()
    if verbose:
        if stage:
            print(f"[模型] 已恢复 checkpoint: stage={stage} epoch={epoch} "
                  f"val_loss={val:.4f} 参数量={model.count_parameters():,}")
        else:
            print(f"[模型] 全新初始化（从零训练） 参数量={model.count_parameters():,}")
    return tok, model, trainer, device


def _get_stale_reason(checkpoint_path):
    """
    检测 checkpoint 是否落后于数据/索引/记忆文件。
    对比 checkpoint 修改时间 vs 以下文件的最新修改时间：
      - train_data/ 目录下 .txt/.json/.md
      - 训练数据/ 目录下学习文件
      - learned_pairs.json
      - external_memory.json
      - tokenizer.json
    返回 (reason_str, latest_mtime) 或 (None, None) 表示未过时。
    """
    if not os.path.exists(checkpoint_path):
        return None, None
    ckpt_mtime = os.path.getmtime(checkpoint_path)
    paths_to_check = []
    # train_data/
    if os.path.isdir(config.TRAIN_FOLDER):
        for root, _dirs, files in os.walk(config.TRAIN_FOLDER):
            for f in files:
                if f.endswith((".txt", ".json", ".md")):
                    paths_to_check.append(os.path.join(root, f))
    # 训练数据/ 学习文件夹
    if os.path.isdir(config.LEARN_FOLDER):
        for root, _dirs, files in os.walk(config.LEARN_FOLDER):
            for f in files:
                if f.endswith((".txt", ".json", ".md")):
                    paths_to_check.append(os.path.join(root, f))
    # 关键索引/记忆文件
    for p in (config.LEARNED_PAIRS_FILE, config.MEMORY_FILE, config.TOKENIZER_FILE):
        if os.path.isfile(p):
            paths_to_check.append(p)
    if not paths_to_check:
        return None, None
    latest_path = max(paths_to_check, key=os.path.getmtime)
    latest_mtime = os.path.getmtime(latest_path)
    if latest_mtime > ckpt_mtime:
        reason = os.path.relpath(latest_path, config.BASE_DIR)
        return reason, latest_mtime
    return None, None


def _recommend_epochs(data_stats, stage="sft"):
    """
    基于数据量自动推荐训练轮数（连续平滑公式，无固定档位）。
    输入: data_stats = {"qa": N, "answers": M, "texts": K}
    策略核心：轮数随数据量对数增长，大模型小数据时自动保守。
      - qa=10  → ~5 轮（防过拟合）
      - qa=50  → ~11 轮
      - qa=200 → ~22 轮
      - qa=1000 → ~39 轮
    同时参考文本量做微调：纯文本多（无 QA）时 LM 轮数增加。
    返回: int 推荐轮数
    """
    qa = data_stats.get("qa", 0)
    texts = data_stats.get("texts", 0)
    if stage == "lm":
        # LM 看纯文本量：对数增长，3–40 轮
        epochs = int(3 + 10 * math.log1p(texts / 30))
        return min(40, max(3, epochs))
    # SFT 看 QA 对数量：对数增长，保底 3 轮封顶 80 轮
    epochs = int(3 + 12 * math.log1p(qa / 50))
    # 额外纯文本奖励（文本量远超 QA 时说明有大量无监督语料）
    if texts > qa * 2:
        epochs = int(epochs * 1.2)
    return min(80, max(3, epochs))


def cmd_tokenize(args):
    tok, *_ = boot(verbose=False)
    texts = args.text or ["你好，安晚"]

    def _norm(s):
        """分词往返比对用：BPE 预分词会丢弃所有空白符（中文聊天场景语义无损），
        故比对前统一去掉空白，避免合法输入被误判为不一致。"""
        return re.sub(r"\s+", "", s or "")

    for t in texts:
        ids = tok.encode(t)
        back = tok.decode(ids)
        print(f"输入: {t}")
        print(f"  ids ({len(ids)}): {ids}")
        print(f"  解码回: {back}")
        if _norm(back) == _norm(t):
            print("  [OK] 往返一致")
        else:
            print("  [!] 往返不一致（非空白部分对不上），词表可能有未登录片段")
            sys.exit(1)
        if len(set(i for i in ids if i not in tok.SPECIAL_TOKENS.values())) == 0:
            print("  [!] 全部落成特殊 token/UNK，检查词表是否覆盖该文本")


def cmd_build_tokenizer(args):
    """从 train_data/ 与 训练数据/ 的真实语料重建 BPE 词表。
    不做这一步，词表会永远停在 5 对种子语料训练出的 50 个 token。"""
    tok, st = build_tokenizer_from_corpus(
        folder=getattr(args, "folder", None), force=getattr(args, "force", False),
        langpack=getattr(args, "langpack", None),
        vocab_size=getattr(args, "vocab_size", None),
        merge_cap=getattr(args, "merge_cap", None))
    if tok is None:
        sys.exit(1)
    if st.get("error") == "no corpus":
        sys.exit(1)
    # 顺带提示 checkpoint 是否需要跟着重训
    if os.path.exists(config.CHECKPOINT_FILE):
        print(f"[提醒] 已存在 checkpoint {config.CHECKPOINT_FILE}。词表变了之后，"
              f"embedding 形状会不匹配——首次续训会走形状迁移（截断/填充），"
              f"旧 embedding 学到的东西会丢一部分。建议词表重建后重训一次。")


def cmd_corpus(args):
    """语料体检：跑一遍解析并报告哪些文件没被用上（不加这步就没法确认语料是否生效）"""
    folder = getattr(args, "folder", None) or config.TRAIN_FOLDER
    pairs, texts, rep, skip = scan_corpus(folder, verbose=True)
    print()
    if skip:
        print(f"  [!] {len(skip)} 个文件没进训练：")
        for fn, why in skip:
            print(f"        {fn} — {why}")
    if not pairs and not texts:
        print(f"  [!] 目录里没有可用的语料。把 .txt/.json/.jsonl/.csv/.md 放进 {folder}")
        sys.exit(1)
    print(f"\n  汇总：问答 {len(pairs)} 条（进 SFT）/ 原文 {len(texts)} 段（进 LM）")
    if len(rep) and len(pairs) < 5:
        print("  [!] 解析出的问答极少，怀疑大部分文件格式没被识别 —— 上面逐个文件看格式标签")


def cmd_word(args):
    """生词本：查 / 加 / 列 / 删"""
    wb = WordBook()
    a = args.action
    if a == "stats":
        print(f"生词本 {wb.stats()} 词条 → {wb.filepath}")
    elif a == "add":
        if not args.word or not args.meaning:
            print("用法：word add <词> <释义> [--pinyin dá]")
            sys.exit(1)
        wb.put(args.word, args.meaning, pinyin=getattr(args, "pinyin", "") or "")
        print(f"已记：{args.word} → {args.meaning}")
    elif a == "search":
        if not args.word:
            print("用法：word search <词>")
            sys.exit(1)
        e = wb.render(args.word)
        print(e if e else f"没查到「{args.word}」")
    elif a == "list":
        items = sorted(wb.words.values(), key=lambda x: x.get("term", ""))
        for it in items[:args.n]:
            print(f"  {it.get('term')} → {it.get('meaning')}")
        print(f"共 {len(items)} 词条")
    elif a == "del":
        k = wb.norm(args.word)
        if k in wb.words:
            del wb.words[k]
            wb.save()
            print(f"已删：{args.word}")
        else:
            print(f"没找到「{args.word}」")
    else:
        print("用法：word stats|add|search|list|del")


def cmd_train(args):
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    stats = pipeline.stats()
    print(f"[数据] qa={stats['qa']} 回复={stats['answers']} 文本={stats['texts']}")
    # 优先使用用户指定的轮数，否则自动推荐
    if args.epochs:
        epochs = args.epochs
        source = "用户指定"
    else:
        if args.lm_only:
            epochs = _recommend_epochs(stats, stage="lm")
        elif args.sft_only:
            epochs = _recommend_epochs(stats, stage="sft")
        else:
            epochs = {"lm": _recommend_epochs(stats, stage="lm"),
                      "sft": _recommend_epochs(stats, stage="sft")}
        source = "自动推荐"
    print(f"[训练] 阶段={'lm' if args.lm_only else ('sft' if args.sft_only else 'lm+sft')}"
          f" epoch={epochs}（{source}）"
          f" device={device}")

    if args.lm_only:
        trainer.train_stage("lm", epochs if isinstance(epochs, int) else epochs["lm"], pipeline)
    elif args.sft_only:
        trainer.train_stage("sft", epochs if isinstance(epochs, int) else epochs["sft"], pipeline)
    else:
        trainer.train_stage("lm", epochs if isinstance(epochs, int) else epochs["lm"], pipeline)
        trainer.train_stage("sft", epochs if isinstance(epochs, int) else epochs["sft"], pipeline)
    print("[训练] 完成")


def _prompt_retrain_or_rematch(reason):
    """
    交互式提示：数据/索引/记忆文件比 checkpoint 新时，让用户选择重新训练或重新匹配。
    非交互环境（如 piped 输入）默认选"重新匹配"。
    返回: "train" | "match" | None（跳过）
    """
    print("=" * 56)
    print("[检测] 数据/索引/记忆文件比 checkpoint 更新")
    print(f"       最新变动来源: {reason}")
    print("请选择处理方式：")
    print("  1) 重新训练 —— 从当前 checkpoint 续训 SFT，纳入新数据（耗时较长）")
    print("  2) 重新匹配 —— 不训练，仅 rebuild 快/慢轨索引与外部记忆向量（快速）")
    print("  3) 跳过 —— 保持现状，直接进入对话")
    try:
        choice = input("输入 1/2/3 (默认=2): ").strip()
    except (EOFError, KeyboardInterrupt):
        return "match"
    if choice == "1":
        return "train"
    elif choice == "3":
        return None
    return "match"  # 默认或空输入


def cmd_chat(args):
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    # 先创建 engine，后续过时检测可以直接对其操作
    engine = InferenceEngine(
        model, tok, device=device, stage=config.STAGE,
        qa_pairs=pipeline.qa_pairs,
        temperature=config.TEMPERATURE, top_p=config.TOP_P,
        max_len=config.GEN_MAX_LEN,
    )
    # 过时检测：数据/索引/记忆文件比 checkpoint 新
    stale_reason, _latest_mtime = _get_stale_reason(config.CHECKPOINT_FILE)
    if stale_reason:
        choice = _prompt_retrain_or_rematch(stale_reason)
        if choice == "train":
            print("[选择] 重新训练：从当前 checkpoint 续训 SFT...")
            sft_epochs = args.epochs if hasattr(args, "epochs") and args.epochs else config.SFT_EPOCHS
            trainer.train_stage("sft", sft_epochs, pipeline)
            print("[训练] 续训完成")
        elif choice == "match":
            print("[选择] 重新匹配：rebuild 索引与记忆向量...")
            engine.rebuild_index(pipeline.qa_pairs)
            print(f"[匹配] 索引 rebuild 完成，语料={len(engine._fast_index)} 条")
            # 外部记忆向量无需重建（add 时实时计算），更新 checkpoint 时间戳避免重复检测
            if os.path.exists(config.CHECKPOINT_FILE):
                os.utime(config.CHECKPOINT_FILE, None)
                print("[匹配] checkpoint 时间戳已更新，下次不再提示")
        else:
            print("[选择] 跳过，保持现状")
    # P1-2：挂载外部长期记忆（事实/日记类；自动持久化到 data/models/external_memory.json）
    engine.memory = ExternalMemory(tok)
    # 生词本（生僻词/外语查询；data/models/wordbook.json，用户自行维护）
    engine.wordbook = WordBook()
    if engine.wordbook.stats():
        print(f"[生词本] 已加载 {engine.wordbook.stats()} 词条")
    # P1-3：恢复上次的跨轮情感状态（含离线时间衰减；无文件则从零）
    engine.load_persisted_emotion()
    # 检索档位：直播弹幕(barrage) / QQ私聊(dm)。同一套图谱，策略按档位切换。
    chat_profile = getattr(args, "mode", None) or config.DEFAULT_PROFILE
    chat_scope = getattr(args, "scope", None) or "user"
    if chat_profile not in config.RETRIEVAL_PROFILES:
        print(f"[!] 未知档位 {chat_profile!r}，回落到 {config.DEFAULT_PROFILE}")
        chat_profile = config.DEFAULT_PROFILE
    print("=" * 56)
    print("安晚 v2 已上线（一体化统一模型 · 单文件版）")
    print("  /reset  重置情感与历史 | /state  查看状态 | /exit 退出")
    print("  /生词 查看 | /生词 加 <词> <释义>")
    print("  /记忆 查看 | /记忆 加 <内容> | /记忆 删 <关键词或id> | /记忆 遗忘 | /记忆 清空 | /记忆 恢复")
    print(f"  /场景 barrage|dm  切换检索档位（当前：{chat_profile}） | 记忆域：{chat_scope}")
    print("=" * 56)

    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save_persisted_emotion()  # 退出兜底：情感状态落盘
            print("\n[已退出] 再见喵~")
            break
        if not q:
            continue
        if q.startswith("/"):
            if q in ("/exit", "/quit", "/byebye"):
                engine.save_persisted_emotion()  # 退出兜底：情感状态落盘
                print("安晚> 再见喵，我会想你的～")
                break
            if q == "/reset":
                engine.reset_emotion()  # P1-3：温和重置（仅清推理侧情感/历史，不删外部记忆与权重）
                print("安晚> 已重置情感与近期对话历史喵～（长期记忆与已学内容保留）")
                continue
            if q == "/场景" or q.startswith("/场景 "):
                arg = q[len("/场景"):].strip()
                if not arg:
                    print(f"[场景] 当前档位：{chat_profile}  "
                          f"（可用：{', '.join(config.RETRIEVAL_PROFILES)}）")
                elif arg in config.RETRIEVAL_PROFILES:
                    chat_profile = arg
                    p = config.RETRIEVAL_PROFILES[arg]
                    print(f"[场景] 已切到 {arg}：注入{p['top_k']}条 / 阈值{p['min_score']} / "
                          f"扩散{'开' if p['spread'] else '关'} / 单条≤{p['max_tokens']}token")
                else:
                    print(f"[场景] 未知档位 {arg!r}，可用：{', '.join(config.RETRIEVAL_PROFILES)}")
                continue
            if q.startswith("/生词"):
                rest = q[len("/生词"):].strip()
                if not rest:
                    items = sorted(engine.wordbook.words.values(),
                                   key=lambda x: x.get("term", ""))
                    if not items:
                        print("[生词本] 还空着喵。用 /生词 加 <词> <释义> 教我。")
                    else:
                        print(f"[生词本] {len(items)} 词条：")
                        for it in items[:20]:
                            print(f"    {it.get('term')} → {it.get('meaning')}")
                elif rest.startswith("加 "):
                    body = rest[2:].strip()
                    parts = body.split(" ", 1)
                    term = parts[0]
                    meaning = parts[1].strip() if len(parts) > 1 else ""
                    meaning = meaning.lstrip("=：: ")
                    if meaning and engine.wordbook.put(term, meaning):
                        print(f"[生词本] 记下啦：{term} → {meaning}")
                    else:
                        print("[生词本] 用法：/生词 加 <词> <释义>")
                else:
                    e = engine.wordbook.render(rest)
                    print(f"[生词本] {e}" if e else f"[生词本] 没查到「{rest}」")
                continue
            if q == "/state":
                emo = engine.emotion_summary()
                mem_stats = engine.memory.stats() if engine.memory else {}
                print(f"[状态] 轮次={engine.rounds} 情感通道={emo}")
                print(f"[状态] 多轮历史={len(engine.history)} 轮 | 外部记忆={mem_stats}")
                print(f"[状态] 情感门控={float(getattr(model,'emotion_gate',1.0)):.3f} "
                      f"(|g|<0.05 说明模型已学会忽略情感) | 情感状态范数="
                      f"{engine.emotion_summary().get('h_norm',0):.2f}")
                print(f"[状态] 场景档位={chat_profile} 记忆域={chat_scope} "
                      f"图谱={engine.memory.graph.stats() if engine.memory else {}}")
                print(f"[状态] 情感持久化={'有' if os.path.isfile(engine.emotion_file) else '无'}"
                      f" | 最近命中: {engine.hit_log[-5:] if engine.hit_log else '无'}")
                continue
            if q == "/train" or q.startswith("/train "):
                # 对话内即时训练：支持 /train 或 /train 50
                rest = q[len("/train"):].strip()
                pipeline2 = DataPipeline(tok, max_len=config.MAX_LEN)
                stats2 = pipeline2.stats()
                if rest:
                    try:
                        epochs = int(rest)
                        source = "用户指定"
                    except ValueError:
                        epochs = _recommend_epochs(stats2, stage="sft")
                        source = "自动推荐（输入无效）"
                else:
                    epochs = _recommend_epochs(stats2, stage="sft")
                    source = "自动推荐"
                print(f"[训练] 即时 SFT 训练 {epochs} 轮（{source}）...")
                trainer.train_stage("sft", epochs, pipeline2)
                engine.update_corpus(pipeline2.qa_pairs)
                print(f"安晚> 训练完成喵～已学习 {stats2['qa']} 条新语料")
                continue
            if q == "/记忆" or q.startswith("/记忆 "):
                print(handle_memory_cmd(engine, q[len("/记忆"):].strip()))
                continue
            if q == "/awil":
                # 切换 AWIL 模块通信流水账开关
                bus = getattr(engine.model, "awil_bus", None)
                if bus is not None:
                    bus.enabled = not bus.enabled
                    state = "开启" if bus.enabled else "关闭"
                    print(f"安晚> AWIL 模块通信流水账已{state}喵~（verbose 模式下回复后会输出模块间消息）")
                else:
                    print("安晚> AWIL 总线未挂载喵")
                continue
            print("安晚> 未知命令，试试 /reset /state /train /记忆 /awil /exit")
            continue

        t0 = time.time()
        # 私聊用固定 scope="user"；弹幕接入时由适配器传入真实发送者 id，
        # 这里是单文件 CLI 的默认值，替换点明确。
        reply, meta = engine.respond(q, scope=chat_scope, profile=chat_profile)
        cost = (time.time() - t0) * 1000
        print(f"安晚> {reply}")
        if args.verbose:
            print(f"      [层={meta['layer']} 意图={meta['intent']} "
                  f"耗时={cost:.0f}ms 情感={meta['emotion']}")
            if meta["router_weights"]:
                top = max(meta["router_weights"], key=meta["router_weights"].get)
                print(f"      [路由主分量={top}({meta['router_weights'][top]}) "
                      f"全权重={meta['router_weights']} 熵={meta['router_entropy']}]")
            print(f"      [记忆注入={meta['memory_hits']} 条 | 历史={meta['history_turns']} 轮]")


def handle_memory_cmd(engine, arg):
    """对话内 /记忆 命令分发：查看 / 加 / 删 / 遗忘 / 清空 / 恢复"""
    mem = engine.memory
    if mem is None:
        return "记忆系统未挂载。"
    if not arg:
        lines = ["[记忆] 最近记忆："]
        items = mem.list(limit=5)
        if not items:
            lines.append("  （暂无记忆，用 /记忆 加 <内容> 记录）")
        for mid, text, weight, hits in items:
            lines.append(f"  #{mid} w={weight} hits={hits} | {text[:40]}")
        st = mem.stats()
        lines.append(f"[记忆] 活跃={st['active']} 归档={st['archived']}（/记忆 恢复 取回）")
        return "\n".join(lines)
    cmd, _, rest = arg.partition(" ")
    cmd = cmd.strip()
    rest = rest.strip()
    if cmd in ("加", "添加", "记", "remember"):
        if not rest:
            return "/记忆 加 <内容> —— 还没给要记的内容喵"
        mid = mem.add(rest, kind="fact")
        return f"已记下喵（#{mid}）：{rest[:40]}{'…' if len(rest) > 40 else ''}"
    if cmd in ("删", "删除", "del", "forget"):
        if not rest:
            return "/记忆 删 <关键词或id> —— 还没给删除目标喵"
        try:
            key = int(rest)
        except ValueError:
            key = rest
        n = mem.forget(key)
        return f"已删除 {n} 条（进入归档，/记忆 恢复 可找回）喵"
    if cmd in ("遗忘", "decay"):
        moved, active, archived = mem.decay()
        return f"平滑遗忘完成：{moved} 条转入归档，活跃 {active}，归档 {archived} 喵"
    if cmd in ("清空", "all"):
        n = mem.forget(None)
        return f"已清空全部 {n} 条记忆（归档中，/记忆 恢复 可找回）喵"
    if cmd in ("恢复", "restore"):
        try:
            idx = int(rest) if rest else 0
        except ValueError:
            idx = 0
        text = mem.restore(idx)
        return f"已恢复：{text[:40]}" if text else "归档为空，没有可恢复的记忆喵"
    if cmd in ("查", "search"):
        if not rest:
            return "/记忆 查 <关键词> —— 没给查询词喵"
        hits = mem.search(rest, top_k=5, min_score=0.0, touch=False)
        if not hits:
            return f"没有命中「{rest}」的记忆喵"
        return "检索命中：\n" + "\n".join(f"  {s:.3f} #{i} | {t[:40]}" for s, t, i in hits)
    return "未知子命令，试试：加 / 删 / 查 / 遗忘 / 清空 / 恢复"


def cmd_mem(args):
    """非交互外部记忆管理：mem stats | add | list | del | search | decay | clear | restore"""
    tok, *_ = boot(verbose=False)
    mem = ExternalMemory(tok)
    if args.action == "stats":
        print(mem.stats())
        return
    if args.action == "add":
        if not args.text:
            print("用法：mem add <内容>")
            return
        mid = mem.add(" ".join(args.text), kind="fact")
        print(f"已添加 #{mid}")
        return
    if args.action == "list":
        items = mem.list(limit=args.n)
        st = mem.stats()
        print(f"[记忆] 活跃={st['active']} 归档={st['archived']}（最近 {len(items)} 条）")
        for mid, text, weight, hits in items:
            print(f"  #{mid} w={weight} hits={hits} | {text[:50]}")
        return
    if args.action == "del":
        if not args.text:
            print("用法：mem del <关键词或id>")
            return
        key = args.text[0]
        try:
            key = int(key)
        except ValueError:
            pass
        print(f"已删除 {mem.forget(key)} 条")
        return
    if args.action == "search":
        if not args.text:
            print("用法：mem search <关键词>")
            return
        hits = mem.search(" ".join(args.text), top_k=args.n, min_score=0.0, touch=False)
        if not hits:
            print("无命中")
            return
        for s, t, i in hits:
            print(f"  {s:.3f} #{i} | {t[:50]}")
        return
    if args.action == "decay":
        moved, active, archived = mem.decay()
        print(f"平滑遗忘：{moved} 条转归档，活跃 {active}，归档 {archived}")
        return
    if args.action in ("clear", "reset"):
        print(f"已清空 {mem.forget(None)} 条")
        return
    if args.action == "restore":
        text = mem.restore(0)
        print(f"已恢复：{text}" if text else "归档为空")
        return
    print("用法：mem {stats|add|list|del|search|decay|clear|restore} [...]")


def cmd_eval(args):
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    trainer.evaluate(pipeline, device=device, n=args.n)


def cmd_moe_report(args):
    """v3 专家化体检：一眼看清「扩了多少参数、真的省了多少算力、情感还是不是完整模型说了算」。"""
    tok = BPEAnwanTokenizer()
    if os.path.exists(config.TOKENIZER_FILE):
        tok.load_vocab(config.TOKENIZER_FILE)
    else:
        print(f"  [!] 无 tokenizer（{config.TOKENIZER_FILE}），用空词表估算结构参数量")
    model = _build_model(tok)
    if os.path.exists(config.CHECKPOINT_FILE):
        ckpt = torch.load(config.CHECKPOINT_FILE, map_location="cpu")
        state = dict(ckpt.get("model_state", {}))
        cur, missing = model.state_dict(), 0
        for name, p in cur.items():
            if name not in state:
                missing += 1
                continue
            cp = state[name]
            with torch.no_grad():
                if p.shape == cp.shape:
                    p.copy_(cp)
                else:
                    dims = [min(a, b) for a, b in zip(p.shape, cp.shape)]
                    sl = tuple(slice(0, d) for d in dims)
                    p[sl].copy_(cp[sl])
        print(f"  [OK] 已加载 checkpoint；新增参数 {missing} 个（半专家输出投影为零初始化 → 行为兼容）")
    else:
        print("  [从零] 无 checkpoint，以下为随机初始化下的结构参数量")
    total = model.count_parameters(trainable_only=False)
    active = model.count_active_parameters()
    print("\n  === v3 专家化体检 ===")
    print(f"  MOE_SCOPE          : {model.moe_scope}")
    if getattr(model, "content_tf", None) is not None:
        print(f"  内容通道           : 双向专家 Transformer {config.CONTENT_TF_LAYERS} 层 × "
              f"{config.CONTENT_TF_EXPERTS} 专家，Top-{config.CONTENT_TF_TOP_K}"
              f"（可一次打包 {config.CONTENT_PACK_MAX_SEGS} 段）")
    elif model.lstm_a_pool:
        print(f"  内容 LSTM-A        : 纯专家 {config.LSTM_A_EXPERTS} 个 × 隐层 {config.LSTM_A_EXPERT_HIDDEN}"
              f"，Top-{config.LSTM_A_TOP_K}")
    elif getattr(model, "ssm_backbone", None) is not None:
        print("  内容通道           : SSM（Mamba2 / S6 回退）")
    else:
        print("  内容 LSTM-A        : 稠密（旧行为）")
    print(f"  检索 TF-1          : 专家 FFN {config.ENC_EXPERTS} 个，Top-{config.ENC_TOP_K}"
          if isinstance(model.tf1, _MoEEncoder) else "  检索 TF-1          : 稠密 FFN（旧行为）")
    print(f"  生成 TF-2          : 专家 FFN {config.NUM_EXPERTS} 个，Top-{config.TOP_K_EXPERTS}")
    if model.lstm_b_semi is not None:
        print(f"  情感 LSTM-B        : 完整模型（隐层 {model.lstm_b_hidden}，持跨轮状态）"
              f" + 半专家 {config.EMO_SEMI_EXPERTS} 个 × {config.EMO_SEMI_EXPERT_HIDDEN}"
              f"，门控上限 {model.emo_semi_gate_max}")
    else:
        print("  情感 LSTM-B        : 仅完整模型（无半专家）")
    print(f"  参数总量           : {total/1e6:.1f}M")
    print(f"  每步激活量         : {active/1e6:.1f}M  ({active/max(total,1)*100:.0f}% —— "
          f"MoE 省下的 {(total-active)/1e6:.1f}M 是未被路由到的专家)")
    vocab = max(model.vocab_size, 8)
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([[BOS_IDX] + [min(2, vocab - 1)] * 4 + [EOS_IDX]])
        model(ids, torch.tensor([[BOS_IDX]]), stage="main")
    usage = model.expert_usage()
    print("  路由分布（本条样例前向后）:")
    for k, v in usage.items():
        print(f"    - {k}: {v}")
    print("  注：随机权重下的分布只用于验证「路由能通、专家都能被走到」，"
          "不代表训练后的分工；训练中请盯 tf1/tf2 是否出现单专家垄断（>0.9 即坍缩）。\n")



def cmd_calibrate_emotion(args):
    """情感超参重标定：在固定语料上统计 SSM 状态范数分布、逐 batch 变化率分布，
    输出建议的 EMOTION_STATE_MAX_NORM（P99）、
    EMOTION_AROUSAL_MARGIN（高低唤醒两组相对分离度中位数）、
    EMOTION_DECAY_DAYS（按状态半衰期拟合）。
    该命令只输出建议值、不自动写 config。
    """
    import numpy as np
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    model.eval()
    
    # 收集统计量
    state_norms = []
    delta_norms = []
    prev_h = None
    arousal_high = []
    arousal_low = []
    n_samples = 0
    max_samples = getattr(args, "max_samples", 200)
    
    print("[标定] 开始收集 SSM 状态统计...")
    print(f"  设备: {device}, 主干: {getattr(config, 'ANWAN_BACKBONE', 'tf')}")
    
    with torch.no_grad():
        for batch in pipeline.sft_batches(min(config.BATCH_SIZE, 8), device):
            if n_samples >= max_samples:
                break
            src = batch["src"]
            lengths = _effective_lengths(src)
            B = src.size(0)
            
            # 前向通过情感通道
            src_emb = model.shared_embedding(src)
            emo_in = model.emb_proj(src_emb)
            
            h0 = torch.zeros(1, B, model.lstm_b_hidden, device=device)
            c0 = torch.zeros(1, B, model.lstm_b_hidden, device=device)
            full_out, (hB, cB) = _packed_lstm(
                model.lstm_b, emo_in, lengths, hx=(h0, c0), total_length=src.size(1)
            )
            h_current = hB.squeeze(0)
            
            # 统计范数
            norms = h_current.norm(dim=-1).cpu().tolist()
            state_norms.extend(norms)
            
            # 统计变化率
            if prev_h is not None and prev_h.size(0) == B:
                deltas = (h_current.detach() - prev_h).norm(dim=-1).cpu().tolist()
                delta_norms.extend(deltas)
            prev_h = h_current.detach().clone()
            
            # 按意图统计唤醒度
            if batch["intents"] is not None:
                intent_names = config.INTENTS
                for i, intent_idx in enumerate(batch["intents"].cpu().tolist()):
                    if 0 <= intent_idx < len(intent_names):
                        intent_name = intent_names[intent_idx]
                        arousal = config.EMOTION_AROUSAL.get(intent_name, 0.35)
                        if arousal >= 0.6:
                            arousal_high.append(norms[i] if i < len(norms) else 0)
                        else:
                            arousal_low.append(norms[i] if i < len(norms) else 0)
            
            n_samples += B
    
    if not state_norms:
        print("[标定] 未收集到足够样本，请检查数据")
        return
    
    state_norms_arr = np.array(state_norms)
    delta_norms_arr = np.array(delta_norms) if delta_norms else np.array([0])
    
    # 计算建议值
    p99_norm = float(np.percentile(state_norms_arr, 99))
    median_delta = float(np.median(delta_norms_arr))
    
    high_mean = float(np.mean(arousal_high)) if arousal_high else float("nan")
    low_mean = float(np.mean(arousal_low)) if arousal_low else float("nan")
    if arousal_high and arousal_low:
        rel_sep = (high_mean - low_mean) / (high_mean + low_mean + 1e-6)
    else:
        rel_sep = 0.0
    
    # 估算半衰期（基于状态变化率衰减）
    if len(delta_norms_arr) > 10:
        # 简单指数拟合：找变化率降到均值一半的位置
        threshold = np.mean(delta_norms_arr) * 0.5
        decay_steps = sum(1 for d in delta_norms_arr if d > threshold)
        # 粗略映射到天（假设每天约 100 步对话）
        estimated_decay_days = max(1.0, decay_steps / 100.0)
    else:
        estimated_decay_days = 7.0
    
    print("\n" + "=" * 60)
    print("[标定结果] 建议配置值（仅建议，不自动写入 config）")
    print("=" * 60)
    print(f"  EMOTION_STATE_MAX_NORM  建议值: {p99_norm:.1f}")
    print(f"    (当前P99状态范数={state_norms_arr.mean():.2f}, "
          f"max={state_norms_arr.max():.2f}, "
          f"std={state_norms_arr.std():.2f})")
    print(f"    说明: 状态范数超过此值时触发重置，建议设为P99以上")
    print()
    high_txt = f"{high_mean:.2f}" if arousal_high else "N/A"
    low_txt = f"{low_mean:.2f}" if arousal_low else "N/A"
    print(f"  EMOTION_AROUSAL_MARGIN  建议值: {rel_sep:.3f}")
    print(f"    (高唤醒均值={high_txt}, 低唤醒均值={low_txt})")
    print(f"    说明: (高唤醒-低唤醒)/(高唤醒+低唤醒) 的中位数分离度")
    print()
    print(f"  EMOTION_DECAY_DAYS      建议值: {estimated_decay_days:.1f}")
    print(f"    (基于状态变化率衰减拟合)")
    print(f"    说明: 情感时间衰减半衰期（天），离线越久情感越平缓")
    print()
    print("应用方式（在 shell 中设置环境变量后启动）：")
    print(f'  export ANWAN_EMOTION_STATE_MAX_NORM="{p99_norm:.1f}"')
    print(f'  export ANWAN_EMOTION_AROUSAL_MARGIN="{rel_sep:.3f}"')
    print(f'  export ANWAN_EMOTION_DECAY_DAYS="{estimated_decay_days:.1f}"')
    print("=" * 60)
    print("[提示] 以上值仅为建议，请根据实际训练效果微调。")
    print("       不修改 config 文件，由环境变量控制。")


def cmd_learn(args):
    """训练数据文件夹系统（老版 A10 的'专门的文件夹功能'）：
    渐进式阅读 BASE_DIR/训练数据/（或 --folder 指定名）下的 .txt/.json/.md，
    每次最多学习 50 条，进度持久化到 anwan_train_progress.json，可重复执行续学；
    --stats 查看进度、--reset 重置、--peek 预览文件。"""
    folder = TrainingDataFolder(
        folder_name=args.folder or LEARN_FOLDER_NAME,
        base_dir=config.BASE_DIR,
    )
    if args.stats:
        print(folder.get_stats())
        return
    if args.reset:
        print(folder.force_relearn(args.reset if args.reset != "__ALL__" else None))
        return
    if args.peek:
        print(folder.peek_file(args.peek))
        return
    pairs, info = folder.learn()
    print(info)
    if pairs:
        print(f"  [learn] 新增 {len(pairs)} 条问答，已并入 {LEARNED_PAIRS_FILE}（train/chat/auto 自动采用；或 --reset 重学）")


def cmd_auto(args):
    """一键启动：自动检测并执行 learn → train → chat 完整链路。
    流程：
    1. 检测 BASE_DIR/训练数据/ 或 train_data/ 是否有新数据 → 自动 learn
    2. 检查 checkpoint 是否存在，以及是否已有训练数据 → 自动 train（若已有 ckpt 且无新增数据则跳过）
    3. 进入 chat 交互对话
    """
    # Step 1: learn（自动学习）
    folder_names = [LEARN_FOLDER_NAME, os.path.basename(TRAIN_FOLDER)]
    any_learned = False
    for fn in folder_names:
        folder = TrainingDataFolder(folder_name=fn, base_dir=config.BASE_DIR)
        pairs, info = folder.learn()
        if pairs:
            any_learned = True
            print(info)
            print(f"  [auto] 文件夹「{fn}」新增 {len(pairs)} 条，已学习完成")
    if not any_learned:
        print("[auto] 暂无新数据待学习（文件夹为空或已全部学完）")

    # Step 2: train（仅在无 checkpoint 或学新数据时触发）
    ckpt_exists = os.path.exists(config.CHECKPOINT_FILE)
    train_needed = (not ckpt_exists) or any_learned
    if train_needed:
        print("[auto] 开始自动训练...")
        tok, model, trainer, device = boot()
        pipeline = DataPipeline(tok, max_len=config.MAX_LEN)
        stats = pipeline.stats()
        if not ckpt_exists:
            print("[auto] 无 checkpoint，执行 LM + SFT 两阶段训练")
            if args.epochs:
                lm_epochs = sft_epochs = args.epochs
                source = "用户指定"
            else:
                lm_epochs = _recommend_epochs(stats, stage="lm")
                sft_epochs = _recommend_epochs(stats, stage="sft")
                source = "自动推荐"
            print(f"[auto] LM {lm_epochs} 轮 / SFT {sft_epochs} 轮（{source}）")
            trainer.train_stage("lm", lm_epochs, pipeline)
            trainer.train_stage("sft", sft_epochs, pipeline)
        else:
            # 有 ckpt 且学了新数据：只做 SFT 微调
            if args.epochs:
                sft_epochs = args.epochs
                source = "用户指定"
            else:
                sft_epochs = _recommend_epochs(stats, stage="sft")
                source = "自动推荐"
            print(f"[auto] 续训 SFT {sft_epochs} 轮（{source}）")
            trainer.train_stage("sft", sft_epochs, pipeline)
        print("[auto] 训练完成")
        # Step 2b: 分化检测与自动追加训练
        report = trainer.lock_and_report(pipeline, device=device)
        entropy = report["global"]["entropy"]
        max_retry = 2
        retry = 0
        while entropy is not None and entropy > 0.7 and retry < max_retry:
            print(f"[auto] 分化不足（熵={entropy:.3f} > 0.7），追加 SFT 训练...")
            add_epochs = args.epochs if args.epochs else 5
            trainer.train_stage("sft", add_epochs, pipeline)
            report = trainer.lock_and_report(pipeline, device=device)
            entropy = report["global"]["entropy"]
            retry += 1
        if entropy is not None and entropy <= 0.7:
            print(f"[auto] 分化达标（熵={entropy:.3f} ≤ 0.7）")
        else:
            print(f"[auto] 分化仍不足（熵={entropy:.3f}），建议扩充语料")
    else:
        print("[auto] checkpoint 已存在且无新数据，跳过训练")

    # Step 3: chat
    print("[auto] 进入对话模式...")
    print()
    cmd_chat(args)




def cmd_baseline(args):
    """基线测试：固定 seed 跑 train --lm-only --epochs 1 + train --sft-only --epochs 1，
    记录 val_seq、chat 中 layer="model" 占比、平均生成长度、moe-report 专家分布，
    写入 data/models/baseline.json。"""
    import random
    seed = getattr(args, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    
    baseline_dir = os.path.join(config.MODEL_DIR, "baseline.json")
    results = {"seed": seed, "backbone": getattr(config, "ANWAN_BACKBONE", "tf"),
               "retrieval_mode": getattr(config, "ANWAN_RETRIEVAL_MODE", "full"),
               "steps": []}
    
    # Step 1: LM only 1 epoch
    print("\n[baseline] Step 1: LM only, 1 epoch")
    trainer.train_stage("lm", 1, pipeline, device)
    lm_val = trainer.history["lm"][-1] if trainer.history["lm"] else {}
    results["steps"].append({"phase": "lm", "epoch": 1, "train_loss": lm_val.get("train"),
                             "val_loss": lm_val.get("val"), "val_seq": lm_val.get("val_seq")})
    
    # 保存中间 checkpoint
    trainer.save_checkpoint(epoch=1, stage="lm", val_loss=lm_val.get("val"))
    
    # Step 2: SFT only 1 epoch
    print("[baseline] Step 2: SFT only, 1 epoch")
    trainer.train_stage("sft", 1, pipeline, device)
    sft_val = trainer.history["sft"][-1] if trainer.history["sft"] else {}
    results["steps"].append({"phase": "sft", "epoch": 1, "train_loss": sft_val.get("train"),
                             "val_loss": sft_val.get("val"), "val_seq": sft_val.get("val_seq")})
    
    # 收集推理指标：正确实例化 InferenceEngine
    print("[baseline] 收集推理指标...")
    model.eval()
    engine = InferenceEngine(model, tok, device=device, stage=getattr(config, "STAGE", "main"))
    engine.memory = ExternalMemory(tok)
    engine.wordbook = WordBook(config.WORDBOOK_FILE)
    
    with torch.no_grad():
        test_texts = ["你好", "今天天气怎么样", "你是谁"]
        gen_lengths = []
        layer_counts = {"model": 0, "memory": 0, "other": 0}
        memory_hits_sum = 0
        for txt in test_texts:
            try:
                reply, meta = engine.respond(txt, scope="user", profile="dm")
            except Exception as e:
                reply, meta = "", {"layer": "other", "memory_hits": 0}
                print(f"  [warn] respond failed: {e}")
            gen_lengths.append(len(reply))
            layer = meta.get("layer", "other")
            if layer in layer_counts:
                layer_counts[layer] += 1
            else:
                layer_counts["other"] += 1
            memory_hits_sum += meta.get("memory_hits", 0)
        
        avg_mem_hits = memory_hits_sum / max(len(test_texts), 1)
        
        # moe-report 专家分布
        usage = model.expert_usage() if hasattr(model, "expert_usage") else {}
    
    results["inference"] = {
        "avg_gen_length": sum(gen_lengths) / max(len(gen_lengths), 1),
        "layer_distribution": layer_counts,
        "expert_usage": str(usage)[:500] if usage else "N/A",
        "avg_memory_hits": avg_mem_hits
    }
    
    # 保存 baseline.json
    os.makedirs(os.path.dirname(baseline_dir), exist_ok=True)
    with open(baseline_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[baseline] 已保存到: {baseline_dir}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_compare(args):
    """对比测试：读 baseline.json 与当前 checkpoint 同类指标，输出对比表；
    额外输出 memory_hits 均值、layer="model"/"memory" 占比。"""
    baseline_dir = os.path.join(config.MODEL_DIR, "baseline.json")
    if not os.path.exists(baseline_dir):
        print(f"[compare] 未找到基线文件: {baseline_dir}")
        print("请先运行 baseline 子命令生成基线数据")
        return
    
    with open(baseline_dir, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    
    # 加载当前模型
    tok, model, trainer, device = boot()
    pipeline = DataPipeline(tok, max_len=config.MAX_LEN,
                            folder_path=getattr(args, "folder", None))
    
    # 收集当前指标
    model.eval()
    current_metrics = {"backbone": getattr(config, "ANWAN_BACKBONE", "tf"),
                       "retrieval_mode": getattr(config, "ANWAN_RETRIEVAL_MODE", "full")}
    
    # 跑验证集计算 loss（与 train_stage 一致的切分逻辑）
    with torch.no_grad():
        # 使用与 train_stage 相同的批次生成（同 batch_size、同 seed=0 默认值）
        all_batches = list(pipeline.sft_batches(config.BATCH_SIZE, device))
        # 与 train_stage 完全一致的验证集切分：前 n_val 个批次为验证集
        n_val = int(len(all_batches) * (1 - config.SFT_TRAIN_RATIO))
        if len(all_batches) >= 4:
            n_val = max(n_val, 1)
        else:
            n_val = 0
        val_batches = all_batches[:n_val]
        
        total_loss = 0.0
        
        for batch in val_batches:
            out = model(batch["src"], batch["tgt_in"],
                        stage=getattr(config, "STAGE", "main"),
                        return_intent=False)
            loss = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)),
                batch["tgt_out"].reshape(-1),
                ignore_index=PAD_IDX
            )
            total_loss += float(loss)
    
    # 用 InferenceEngine 收集推理指标
    engine = InferenceEngine(model, tok, device=device, stage=getattr(config, "STAGE", "main"))
    engine.memory = ExternalMemory(tok)
    engine.wordbook = WordBook(config.WORDBOOK_FILE)
    
    test_texts = ["你好", "今天天气怎么样", "你是谁"]
    gen_lengths = []
    layer_counts = {"model": 0, "memory": 0, "other": 0}
    memory_hits_sum = 0
    
    with torch.no_grad():
        for txt in test_texts:
            try:
                reply, meta = engine.respond(txt, scope="user", profile="dm")
            except Exception as e:
                reply, meta = "", {"layer": "other", "memory_hits": 0}
                print(f"  [warn] respond failed: {e}")
            gen_lengths.append(len(reply))
            layer = meta.get("layer", "other")
            if layer in layer_counts:
                layer_counts[layer] += 1
            else:
                layer_counts["other"] += 1
            memory_hits_sum += meta.get("memory_hits", 0)
    
    n_val = max(len(val_batches), 1)
    current_metrics["val_loss_avg"] = total_loss / n_val
    current_metrics["avg_gen_length"] = sum(gen_lengths) / max(len(gen_lengths), 1)
    current_metrics["layer_distribution"] = layer_counts
    current_metrics["avg_memory_hits"] = memory_hits_sum / max(len(test_texts), 1)
    
    # 输出对比表
    print("\n" + "=" * 70)
    print("[对比报告] baseline vs current")
    print("=" * 70)
    print(f"{'指标':<25s} {'基线':<20s} {'当前':<20s}")
    print("-" * 70)
    
    # 对比训练指标
    for step in baseline.get("steps", []):
        phase = step.get("phase", "")
        print(f"\n[{phase.upper()} epoch {step.get('epoch', '?')}]")
        for key in ["train_loss", "val_loss", "val_seq"]:
            bl_val = step.get(key)
            cv = current_metrics.get(key)
            print(f"  {key:<20s} {str(bl_val):<20s} {str(cv):<20s}")
    
    # 对比推理指标
    print(f"\n[推理指标]")
    bl_infer = baseline.get("inference", {})
    for key in ["avg_gen_length", "avg_memory_hits"]:
        bl_val = bl_infer.get(key, "N/A")
        cv = current_metrics.get(key, "N/A")
        print(f"  {key:<20s} {str(bl_val):<20s} {str(cv):<20s}")
    
    print("\n" + "=" * 70)
    print("[提示] 当前指标基于当前 checkpoint 在验证集上的表现。")
    print("       基线数据来自 cmd_baseline 记录的固定 seed 运行结果。")
    print(f"       当前使用主干: {current_metrics.get('backbone', 'unknown')}")
    print("=" * 70)


def main():
    # 目录体检：启动先核对关键文件落点，放错位置时提醒并自动归位（在任何子命令前执行）
    check_and_relocate()

    parser = argparse.ArgumentParser(
        prog="anwan_single",
        description="安晚一体化统一模型 v2 单文件版：tokenize / train / chat / eval / learn")
    sub = parser.add_subparsers(dest="cmd")

    p_tok = sub.add_parser("tokenize", help="分词往返检查")
    p_tok.add_argument("--text", nargs="*", default=None)

    p_vocab = sub.add_parser("build-tokenizer",
                             help="从 train_data/ 真实语料重建 BPE 词表（加语料前必做）")
    p_vocab.add_argument("--folder", type=str, default=None, help="额外的语料目录")
    p_vocab.add_argument("--force", action="store_true", help="已存在词表时强制重建")
    p_vocab.add_argument("--langpack", type=str, default=None,
                         help="语言包目录（可含多语种子目录，训练后输出分语种覆盖报告）")
    p_vocab.add_argument("--vocab-size", type=int, default=None,
                         help="词表上限（如 40000；会写入 config）")
    p_vocab.add_argument("--merge-cap", type=int, default=None,
                         help="BPE 合并次数上限（决定词表实际能涨到多大）")

    p_corpus = sub.add_parser("corpus", help="语料体检：看哪些文件被用上、哪些被跳过")
    p_corpus.add_argument("--folder", type=str, default=None, help="语料目录（默认 train_data/）")

    p_word = sub.add_parser("word", help="生词本：生僻词/外语查询（stats/add/search/list/del）")
    p_word.add_argument("action", type=str, help="stats | add | search | list | del")
    p_word.add_argument("word", nargs="?", default=None, help="待查/待加的词")
    p_word.add_argument("meaning", nargs="?", default=None, help="释义（add 时用）")
    p_word.add_argument("--pinyin", type=str, default="", help="拼音（可选）")
    p_word.add_argument("-n", type=int, default=30, help="list 显示条数")

    p_train = sub.add_parser("train", help="分段训练（LM 预训练 + SFT 调优）")
    p_train.add_argument("--lm-only", action="store_true")
    p_train.add_argument("--sft-only", action="store_true")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--folder", type=str, default=None,
                         help="外部语料目录（默认 train_data/）")

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("-v", "--verbose", action="store_true",
                        help="显示命中层 / 意图 / 路由 / 耗时调试信息")
    p_chat.add_argument("--epochs", type=int, default=None,
                        help="即时训练默认轮数（对话内 /train 使用，未指定时自动推荐）")
    p_chat.add_argument("--mode", type=str, default=None,
                        choices=list(config.RETRIEVAL_PROFILES.keys()),
                        help="检索档位：barrage=直播弹幕(高压低延迟) / dm=私聊(低频高粘)")
    p_chat.add_argument("--scope", type=str, default=None,
                        help="记忆归属域 id（弹幕接入时传发送者 id，用于隔离）")

    p_eval = sub.add_parser("eval", help="评测基线")
    p_eval.add_argument("--n", type=int, default=20)

    p_auto = sub.add_parser("auto", help="一键启动：自动检测→learn→train→chat")
    p_auto.add_argument("--epochs", type=int, default=None, help="训练轮数（默认 LM/SFT 各 30）")
    p_auto.add_argument("-v", "--verbose", action="store_true",
                        help="对话模式显示命中层 / 意图 / 路由 / 耗时调试信息")
    p_auto.add_argument("--mode", type=str, default=None,
                        choices=list(config.RETRIEVAL_PROFILES.keys()),
                        help="检索档位：barrage=直播弹幕 / dm=私聊")
    p_auto.add_argument("--scope", type=str, default=None,
                        help="记忆归属域 id（弹幕接入时传发送者 id）")

    p_learn = sub.add_parser("learn", help="训练数据文件夹系统：渐进式学习/进度查询/重置/预览")
    p_learn.add_argument("--folder", type=str, default=None,
                         help="文件夹名（默认：训练数据，位于 BASE_DIR 下）")
    p_learn.add_argument("--stats", action="store_true", help="查看学习进度统计")
    p_learn.add_argument("--reset", nargs="?", const="__ALL__", default=None,
                         help="重置学习进度（不带参数重置全部，或指定文件名）")
    p_learn.add_argument("--peek", type=str, default=None, help="预览指定文件的前 5 条问答")

    p_moe = sub.add_parser(
        "moe-report", help="v3 专家化体检：总/激活参数量、情感半专家门控、各专家池路由分布")
    p_cal = sub.add_parser("calibrate-emotion",
                          help="情感超参重标定：统计SSM状态分布，输出建议的EMOTION_STATE_MAX_NORM等")
    p_cal.add_argument("--folder", type=str, default=None, help="语料目录")
    p_cal.add_argument("--max-samples", type=int, default=200, help="最大采样数")
    p_base = sub.add_parser("baseline",
                           help="基线测试：固定seed跑1轮LM+SFT，记录指标到baseline.json")
    p_base.add_argument("--seed", type=int, default=42, help="随机种子")
    p_base.add_argument("--folder", type=str, default=None, help="语料目录")
    p_cmp = sub.add_parser("compare",
                          help="对比测试：读baseline.json与当前checkpoint指标对比")
    p_cmp.add_argument("--folder", type=str, default=None, help="语料目录")
    p_mem = sub.add_parser("mem", help="外部长期记忆管理（P1-2）：stats/add/list/del/search/decay/clear/restore")
    p_mem.add_argument("action", type=str, help="stats | add | list | del | search | decay | clear | restore")
    p_mem.add_argument("text", type=str, nargs="*", default=None, help="内容/关键词/id")
    p_mem.add_argument("--n", type=int, default=10, help="list/search 返回条数")

    args = parser.parse_args()
    if args.cmd == "tokenize":
        cmd_tokenize(args)
    elif args.cmd == "corpus":
        cmd_corpus(args)
    elif args.cmd == "word":
        cmd_word(args)
    elif args.cmd == "build-tokenizer":
        cmd_build_tokenizer(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "chat":
        cmd_chat(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "learn":
        cmd_learn(args)
    elif args.cmd == "mem":
        cmd_mem(args)
    elif args.cmd == "moe-report":
        cmd_moe_report(args)
    elif args.cmd == "calibrate-emotion":
        cmd_calibrate_emotion(args)
    elif args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "auto":
        cmd_auto(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
