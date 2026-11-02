# Changelog

## 2026-11-02

### 前置通用路由

- 中央 router 改为进模型就拍板：用输入摘要、意图 embedding、上一轮 PAD 出 4 通道权重，再决定是否跑内容 / 检索。
- 情感 LSTM-B 与生成 TF-2 始终在线；内容、检索在推理期低于 `ROUTER_SKIP_THRESHOLD` 可跳过。训练期四路全开，梯度能回到 router。
- 分化锁定按意图各存一张 top-2 掩码（情感+生成强制常驻），不再用全局同一张 mask。
- 旧 checkpoint 的 `module_projs` / `router_net` 形状对不上，加载时走形状适配丢弃，需续训。

### 上下文上限

- 按 800M、2 层内容 + 5 层检索 + 4 层解码能稳住的智力，输入从 80 拉到 `MAX_LEN=256`，生成从虚标 1024 收到 `GEN_MAX_LEN=256`，训练目标同步 `TGT_MAX_LEN=256`。
- 历史 6 轮 / 120 token，记忆单条 48，打包段数 6；SFT/LM 按 batch 内最长 padding，不再每条补满 256。

## 2026-10-31

### 内容通道

- 默认主干从 SSM 改为双向专家 Transformer（`ANWAN_BACKBONE=tf`）：标准 `MultiheadAttention`，手机端可跑；FFN 为 4 专家 Top-2。
- 内容通道一次可吃多段文本：`<BOS> t1 <SEP> t2 ...`，段 embedding 区分来源；LM/SFT 以 `CONTENT_PACK_PROB=0.4` 掺入，最多 `CONTENT_PACK_MAX_SEGS=4` 段。
- SSM / LSTM-A 专家池仍可通过 `ANWAN_BACKBONE=ssm|lstm` 回退。

## 2026-10-30

### 修复

- `config` 改为自动收集全部大写常量，补上 `LEARN_FOLDER` / `PROGRESS_FILE` 等原先手工列表漏掉的键。
- 情感通道不再把 512 维输入硬塞进 1536 维 SSM；跨轮状态始终由 LSTM-B 的 `(h, c)` 持有，形状与 packed LSTM / 持久化文件一致。
- Mamba2 导入改为 `mamba_ssm.modules.mamba2`（旧路径 `models.mixer_seq_simple` 不存在）；原生 mixer 返回值不再按二元组解包。
- RMSNorm 回退去掉重复归一化。
- S6 因果卷积改为左侧 pad、`padding=0`，避免 `Conv1d` 默认双边 pad 把序列拉长；门控 `gate` 重新参与输出。
- `pack_padded_sequence` 的 lengths 固定走 CPU，避免 CUDA tensor 触发报错。
- 质量门改为「字母 + CJK」占比，不再因纯外文回复被误判失败。
- `calibrate-emotion` 用 LSTM-B 状态范数，高低唤醒均值未定义时不再 NameError。
- `compare` 的 `ignore_index` 改为 `PAD_IDX`。
- 意图头默认维度与 14 类 `INTENTS` 对齐；过时的 240M / 12 意图注释已改。

### 文档

- 用法、环境、目录、规格从 `a12.py` 文件头迁到 `README.md`。
- 新增 `requirements.txt`。
