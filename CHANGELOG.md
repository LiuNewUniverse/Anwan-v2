# Changelog

## 2026-11-05

### 1800M 规格

- 总量从约 1500M 拉到约 1800M。宽度仍 1536，增量给专家 FFN 与情感隐层：生成路由 1920 / 共享 3840，检索与内容路由 1280 / 共享 2560，LSTM-B 3600→4096，半专家隐层 1200→1365。通道比例仍约 生成 45% / 检索 26% / 内容 19% / 情感 10%。
- 旧 checkpoint 的 FFN 宽度 / LSTM-B 隐层对不上，加载时走形状适配截断或丢弃，需续训。

### DeepSeek 式细粒度分化池

- `_MoELayer` 补齐细粒度切分：`MOE_GRANULARITY=2`，生成/检索/内容路由专家 8→16、Top-2→Top-4，路由 FFN 减半；共享专家保持原宽度、全体 token 必跑、不进 gate。
- 无辅助损失均衡：每专家 `expert_bias`（buffer，无梯度）只加在 top-k 选择分数上，组合权重用原始 gate scores；训练中按 `sign(k/N - load) * γ` 更新。GShard aux 默认关闭（`ANWAN_MOE_AUX_WEIGHT=0`），需要时可再打开。
- 路由分数可选 softmax / sigmoid（`ANWAN_MOE_GATE_SCORE`）。`moe-report` 打印切分倍数、bias γ、aux 权重和各层 bias。
- 旧 checkpoint 的专家数 / FFN 宽度 / `gate.bias` 对不上，加载时走形状适配截断或丢弃，需续训。

## 2026-11-02

### 1500M + DeepSeek 式共享专家

- 总量从约 800M 拉到约 1500M。宽度仍 1536，增量给专家池：生成 6 层 ×（8 路由 + 2 共享）、检索 6 层 ×（8 路由 + 1 共享）、内容 4 层 ×（8 路由 + 2 共享）、情感半专家 4 个。通道比例约 生成 45% / 检索 26% / 内容 19% / 情感 10%。
- `_MoELayer` 加入共享专家：每层 1~2 个始终在线的 FFN，路由专家仍 Top-2；aux_loss 只打在路由专家上。共享专家参数始终计入激活量。
- 旧 checkpoint 的专家数 / 层数对不上，加载时走形状适配截断或丢弃，需续训。

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
