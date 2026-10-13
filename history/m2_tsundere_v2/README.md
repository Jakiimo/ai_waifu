# AI Waifu

基于 Qwen 和 LoRA 的中文角色对话学习项目。项目目标是实践可复现微调，并通过强 Prompt 基线判断微调的额外收益。

当前角色是“凛音”，嘴硬心软的傲娇型，见 [角色卡](docs/凛音_角色卡.md)。M2 主流程已经统一到 `m2-tsundere-v2`：360 条训练样本／120 个场景族（含60条带前文样本）、24 个开发场景、60 个最终测试场景。当前 [30条台词抽查](results/m2/review_samples.md) 与 [开发集对照](results/m2/prompt_comparison.md) 都对应这个版本。

M1 本地推理、小样本训练和新进程重载已通过。M2 新版数据检查与9题×3提示词的真实原模型对照已执行；原模型角色感和部分技术回答仍不合格，提示词选择暂定，工程包尚未冻结；30条人工抽查已记录（18条pass、12条revise），台词修订及全量审核未完成。M3 正式角色训练、M4 Demo、M5 最终评测尚未执行。

本次版本混用的原因与修复见 [M2一致性排查](docs/M2_一致性排查.md)。旧版完整准备包保存在 `history/m2_persona_v1/`；`data/m2_tsundere_v2/` 仅保留首次候选快照，主流程现在读取 `data/m2/`。

最新台词偏好与审核结果见 [人工抽查反馈](docs/M2_人工抽查反馈.md)。

实际记录见 [M1 验证记录](docs/M1_验证记录.md) 和 [M2 验证记录](docs/M2_验证记录.md)。

## 从这里开始

- [M1 入门教程](docs/M1_入门教程.md)：适合没有微调经验的读者，按步骤运行并解释原理。
- [M2 入门教程](docs/M2_入门教程.md)：理解训练/开发/测试、数据增强、强 Prompt 基线与冻结。
- [M2 数据说明](data/m2/DATA_CARD.md)：合成来源、360 条与 120 个场景的区别、多轮监督和人工审核状态。
- [开发集 Prompt 对照](results/m2/prompt_comparison.md) 与 [选择记录](results/m2/prompt_decision.md)：真实输出和失败分析。
- [产品需求文档](PRD.md)：第一版范围、里程碑和验收条件。
- [M1 教学训练配置](configs/m1.json) 与 [M2 角色实验配置](configs/m2.json)：不同阶段分别记录，M1 保留历史验证内容。
- [教学数据说明](data/README.md)：12 条示例的来源与用途。
- [M1 实现](scripts/m1.py)：复用 Trainer 与 PEFT 的流程。
- [可选学习 Notebook](train.ipynb)：逐格执行相同步骤，终端路线不要求使用 Notebook。

## 本机命令

在项目目录运行，首次使用先阅读教程：

```bash
.venv/bin/python scripts/m1.py doctor
.venv/bin/python scripts/m1.py download
.venv/bin/python scripts/m1.py baseline
.venv/bin/python scripts/m1.py inspect-data
.venv/bin/python scripts/m1.py train
.venv/bin/python scripts/m1.py verify
.venv/bin/python scripts/m1.py status
```

训练命令不会覆盖已有 Adapter。若当前实验已经通过，初次学习可以先阅读记录并复跑 doctor、baseline、inspect-data 与 verify，理解之后再建立下一轮实验。

M2 当前检查命令：

```bash
.venv/bin/python prepare_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/m2.py status
```

新版开发对照已经保存，`dev-probe` 会核对同版本缓存，避免重复生成。`freeze` 目前会拒绝暂定的提示词决策；先处理开发问题与人工台词意见，再建立可冻结版本。M1 的 `train` 命令只训练12条教学样本，不能用于360条角色数据。

本机采用 PyTorch MPS、Transformers 和 PEFT 的普通 LoRA，不是 4-bit QLoRA。`.venv` 使用 Python 3.12；完整实测依赖在 `requirements-lock.txt`。

## 开源复用

| 组件 | 来源 | 项目工作 |
|---|---|---|
| 基础模型、分词与聊天模板 | Qwen/Qwen2.5-1.5B-Instruct | 模型版本固定、角色提示词和本地加载 |
| 推理与训练循环 | Hugging Face Transformers | 数据接入、训练配置、运行记录 |
| LoRA 实现与权重管理 | Hugging Face PEFT | 适配模块选择、参数变化和重载验证 |
| 教学示例与验证流程 | 本项目，AI 辅助开发 | 标签检查、流程验收、入门教程 |

参考官方模型卡、Trainer、PEFT 文档实现；未声称 fork 了完整 AI Waifu 应用，也未声称从零实现训练算法。Qwen 模型卡标注 Apache-2.0；各开源库许可证以对应发行版本为准。

## 结果与文件

- `models/`：基础模型与固定版本记录，只保存在本机。
- `outputs/`：Adapter 与临时训练输出，只保存在本机。
- `results/m1/`：环境、原模型回答、标签检查、训练和重载记录。
- `.venv/`：项目独立环境。

大文件、缓存与秘钥已配置忽略。M1 示例不含真实用户资料，训练未建立长期记忆、工具调用或聊天界面。

12 条示例和 8 次参数更新只用于 M1 流程验证，不能证明质量提升。
