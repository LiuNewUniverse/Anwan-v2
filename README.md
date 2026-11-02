# Anwan-v2

安晚一体化统一模型 v2：单文件自包含（训练 + 对话）。规格约 800M 参数。默认内容通道为双向专家 Transformer（标准注意力，手机端可用），一次可打包多段文本；情感通道始终走 LSTM-B 以保持跨轮 `(h, c)`。中央路由前置：看输入摘要、意图、上一轮 PAD 后决定跑哪些通道，情感与生成永不跳过。SSM / LSTM-A 仍可经环境变量回退。

## 环境

- Python 3.10+
- 必选：`torch>=2.1.0`（CPU 可跑，CUDA 用于 AMP / 大模型训练）
- 可选：`mamba-ssm`、`causal-conv1d`（仅 `ANWAN_BACKBONE=ssm` 时需要原生 Mamba2；缺省走纯 PyTorch S6）
- 标定命令 `calibrate-emotion` 需要 `numpy`（多数 torch 环境已带）

```bash
pip install -r requirements.txt
```

GPU 训练建议 Ampere 及以上、显存 16GB+。Orin Nano 8G 可把 `CKPT_SAVE_SHADOWS = False`，并视情况关 AMP 以外的影子权重。

## 目录

脚本启动时自动选可写 `BASE_DIR`（脚本目录 / Android 0 目录 / 文档目录 / 当前目录）。

| 路径 | 用途 |
|------|------|
| `data/models/tokenizer.json` | BPE 词表 |
| `data/models/anwan_v2.pt` | checkpoint |
| `data/models/wordbook.json` | 生词本 |
| `data/models/external_memory.json` | 长期记忆 |
| `data/models/emotion_state.pt` | 跨轮情感状态 |
| `train_data/` | 训练语料（`.txt` / `.json` / `.md`） |
| `训练数据/` | `learn` 渐进阅读 |

## 用法

```bash
# 1. 语料放进 train_data/
# 2. 建词表（不做则保底约 50 token，新语料全落 UNK）
python a12.py build-tokenizer

# 多语
python a12.py build-tokenizer --langpack langdata --vocab-size 22000 --merge-cap 20000

# 3. 训练（LM + SFT）
python a12.py train

# 4. 对话
python a12.py chat
```

子命令：`tokenize` | `build-tokenizer` | `train` | `chat` | `eval` | `learn` | `mem` | `word` | `auto` | `moe-report` | `calibrate-emotion` | `baseline` | `compare`。各子命令参数用 `-h` 查看。

一键：`python a12.py auto`（learn → train → chat）。

## 环境变量

| 变量 | 取值 | 说明 |
|------|------|------|
| `ANWAN_BACKBONE` | `tf` / `ssm` / `lstm` | 内容通道主干，默认 `tf`（双向专家 Transformer） |
| `ANWAN_RETRIEVAL_MODE` | `full` / `facts_only` | 记忆检索模式 |
| `ANWAN_EMO_DIR_NORM` | `raw` / `std` | 情绪方向 loss 是否对变化率做 z-score |
| `ANWAN_MOE_SCOPE` | `all` / `decoder` | 全域专家化或仅 decoder MoE |
| `ANWAN_EMOTION_STATE_MAX_NORM` | float | 情感状态范数上限，越界重置 |
| `ANWAN_EMOTION_AROUSAL_MARGIN` | float | 高低唤醒相对分离度边距 |
| `ANWAN_EMOTION_DECAY_DAYS` | float | 离线情感衰减半衰期（天） |

## 规格（不降配）

- Embedding 1536；内容通道默认 2 层双向 MoE Transformer（4 专家 Top-2，FFN 1536）
- 一次最多打包 6 段文本（`<SEP>` 分隔 + 段 embedding）；LM/SFT 以 0.4 概率掺入
- LSTM-B hidden 3600；情感 512 维 + PAD 三维（愉悦 / 唤醒 / 支配）
- TF-1：5 层；TF-2：4 层 MoE decoder（4 专家 Top-2）
- 前置通用路由：输入摘要 + 意图 embedding + PAD → 4 通道权重；推理时内容/检索低于 `ROUTER_SKIP_THRESHOLD=0.12` 可跳过；按意图锁定 top-2（情感+生成强制常驻）
- 词表 32000；输入 `MAX_LEN=256`、训练/生成 `TGT_MAX_LEN=GEN_MAX_LEN=256`（2+5+4 层能稳住的读写上限）；历史 6 轮 / 120 token，记忆单条 48 token
- 意图 14 类（含 `lang_learning` / `rare_word`）
- 回退：`ANWAN_BACKBONE=ssm` 或 `lstm`（LSTM-A hidden 8192 / 专家隐层 4096）

更新日志见 `CHANGELOG.md`。
