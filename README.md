# AI Waifu

基于 Qwen 和 LoRA 的中文角色对话学习项目。项目目标是实践可复现微调，并通过强 Prompt 基线判断微调的额外收益。

当前角色是“凛音”，嘴硬心软的傲娇型，见 [角色卡](docs/凛音_角色卡.md)。M2 主流程已经统一到 `m2-tsundere-v3`：360 条训练样本／120 个场景族（含60条带前文样本）、24 个开发场景、60 个最终测试场景。当前 [30条台词抽查](results/m2/review_samples.md) 与 [开发集对照](results/m2/prompt_comparison.md) 都对应这个版本。

M1本地推理、小样本训练和新进程重载已通过。M2已完成v3数据与实验准备：当前30条抽查全部pass，23项检查通过，固定9题×3种Prompt的27条实际回答已保存，本轮详细基线选定strong_examples并冻结。原模型的角色和技术失败保留，不宣称模型质量合格或微调改善。M3已实际训练360条实例×2轮、完成180次更新和新进程重载；开发loss从4.1188降到3.3701，Adapter约4.17MiB。学习者亲手完成的m3_learning_01也已重载通过并保存18条开发对照。M4推理与Gradio界面已实现并完成开发检查；学习者已反馈教程体验完成。M5正式回答240/240、生成失败0；按用户委托，Codex已统一评分120对，来源、每对理由和旧评分备份均保留，本轮不是人工盲评。详细Prompt主对照LoRA为17胜、11平、32负，短Prompt为12胜、10平、38负，未支持整体质量改善。见[结果解读](results/m5_learning_01/结果解读.md)及[完整报告](results/m5_learning_01/report.md)。

本次版本混用的原因与修复见 [M2一致性排查](docs/M2_一致性排查.md)。旧版完整准备包保存在 `history/m2_persona_v1/` 和 `history/m2_tsundere_v2/`；`data/m2_tsundere_v2/` 仅保留首次候选快照，主流程现在读取 `data/m2/`。

M6小样本20对、5次更新与重载已由学习者完成，6题观察未显示明确整体改善。扩展后420条候选、失败0，助手审核210题并冻结179组train/6组dev；训练chosen原选26条、AI修订153条，开发无修订。学习者已完成正式DPO：179对、1轮45次更新，112个policy LoRA张量变化，reference未变，新进程重载最大logit差0.0；开发观察未支持明确整体改善。同起点chosen-only SFT、最终质量对照与M7尚未完成，操作见[M6第四步教程](docs/M6_第四步_同起点训练对照.md)。

用户要求的12条台词已直接修改，同类38个学习/技术场景也已统一口吻，共影响150条训练实例；其余70个场景族保留。见 [12条修改前后](results/m2/revised_12_samples.md)。

最新台词偏好与审核结果见 [人工抽查反馈](docs/M2_人工抽查反馈.md)。

实际记录见 [M1 验证记录](docs/M1_验证记录.md) 、[M2 验证记录](docs/M2_验证记录.md) 和 [M3 验证记录](docs/M3_验证记录.md)。

## 从这里开始

- [M6 入门教程](docs/M6_入门教程.md)：从当前SFT生成同题两条候选，助手审核偏好，再逐步进入DPO；不自动代跑正式实验。
- [M6 第二步教程](docs/M6_第二步_DPO小样本教程.md)：查看实际偏好对，亲手训练20对、5次更新并观察6条开发回答。
- [M6 第三步教程](docs/M6_第三步_扩展偏好数据.md)：保留第一批候选，扩展训练与开发采样；助手审核后再冻结正式对照。
- [M6 第四步教程](docs/M6_第四步_同起点训练对照.md)：实际179/6数据冻结，同起点两条训练、15步开发观察与独立重载。
- [M6 扩展审核结果](results/m6_preparation/reviews/expansion_report.md)：全部AI反馈，原候选、修订、拒收、数据覆盖与来源局限。
- [M5 结果解读](results/m5_learning_01/结果解读.md)：本轮评分已委托AI完成，直接理解胜平负、任务退化与证据局限。
- [M5 入门教程](docs/M5_入门教程.md)：四组公平对照与原人工评分路线；本轮AI来源见[评分方法](results/m5_learning_01/ai_review/评分方法.md)。
- [M4 入门教程](docs/M4_入门教程.md)：先在终端调用reply，再亲手启动网页，解释模型加载、推理和会话历史。
- [M3 逐步学习](tutorial.md)：先理解配置与代码，再亲手启动独立实验；训练前和每轮结束生成开发回答。
- [M1 入门教程](docs/M1_入门教程.md)：适合没有微调经验的读者，按步骤运行并解释原理。
- [M3 入门教程](docs/M3_入门教程.md)：正式训练、新进程重载与开发观察，逐步解释原理。
- [M3 开发对照](results/m3_tsundere_v3_r8_e2/development_comparison.md)：真实原模型与LoRA回答，保留失败。
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
.venv/bin/python scripts/m2.py dev-probe
.venv/bin/python scripts/m2.py status
```

准备包已冻结，prepare_data校验文件，dev-probe核对并复用冻结回答，freeze校验已有清单，均不覆盖记录。当前30条示范全部pass；具体学习步骤及只读标签练习见M2教程第5节。M1的train命令只处理12条教学样本，M3使用下方独立训练入口。

M3本轮已训练完成，可以按教程检查：

```bash
.venv/bin/python scripts/m3.py inspect-data
.venv/bin/python scripts/m3.py train
.venv/bin/python scripts/m3.py verify
.venv/bin/python scripts/m3.py dev-compare
.venv/bin/python scripts/m3.py status
```

同一run的train不会覆盖或重复训练，dev-compare会复用当前开发观察；verify实际从磁盘加载模型。正式配置在configs/m3.json，独立产物在outputs/m3_tsundere_v3_r8_e2/，记录在results/m3_tsundere_v3_r8_e2/。

亲手启动新一轮并观察中途回答：

```bash
.venv/bin/python scripts/m3.py inspect-data --config configs/m3_learning.json
.venv/bin/python scripts/m3.py train --config configs/m3_learning.json
```

学习者已亲手完成`m3_learning_01`，保留旧实验。训练前及每轮结束的6条固定开发回复保存在`results/m3_learning_01/development_timeline.md`，重载通过，18条最终开发对照见`results/m3_learning_01/development_comparison.md`。选择该配置后，status、verify等命令也需带相同的`--config`参数。

M4先检查产物、生成一句回复，再启动网页：

```bash
.venv/bin/python demo.py --check
.venv/bin/python demo.py --message "今天把房间收拾干净了。"
.venv/bin/python demo.py
```

网页默认`http://127.0.0.1:7860`，终端需保持运行。支持当前会话聊天、原模型/LoRA切换、新会话和独立同题比较。训练入口与训练数据不参与Demo推理。聊天使用short Prompt，同题比较两边使用相同strong_examples Prompt；超长历史按完整轮次裁剪并提示，当前问题过长则拒绝截断。

界面依赖在当前环境已安装，换环境运行`.venv/bin/python -m pip install -r requirements-m4.txt`；M4完整依赖快照为`requirements-m4-lock.txt`，原训练环境版本保持不变。实际开发检查见[M4验证记录](docs/M4_验证记录.md)，学习者按教程完成网页体验。

M5先停止仍在运行的M4网页，再依次执行：

```bash
.venv/bin/python scripts/m5.py prepare
.venv/bin/python scripts/m5.py generate
.venv/bin/python scripts/m5.py blind
.venv/bin/python scripts/m5_review.py
```

prepare只核对输入并固定条件；generate由学习者执行60题×4组，共240回答，逐条保存。中断用generate --resume继续。匿名评分网页默认`http://127.0.0.1:7862`，不加载模型、不默认填分；120对覆盖四组回答，可分批保存。读完回答再评分，暂不打开原始回答或私有身份映射。

以上是原人工评分路线；本轮用户已将评分委托AI且全部完成，重建现有报告运行`.venv/bin/python scripts/m5_ai_report.py`，不能把AI分数登记成“作者本人”人工评分。结果在`results/m5_learning_01/`，详见[M5验证记录](docs/M5_验证记录.md)。

M6使用新环境，只先采集小批训练候选：

```bash
.venv-m6/bin/python scripts/m6.py status
.venv-m6/bin/python scripts/m6.py collect --split train --limit 20
.venv-m6/bin/python scripts/m6.py review-pack --split train
```

20题×两次采样得到40条SFT候选，不更新LoRA。候选中断加`--resume`；没有真实审核不能导出chosen/rejected。当前第一批已审核为20对，2对直接选择原候选、18对明确记录AI修订chosen，原始40条保存。新环境已安装，完整版本在requirements-m6-lock.txt，旧环境和模型保留。

第二步由学习者依次执行（train会开始实际更新）：

```bash
.venv-m6/bin/python scripts/m6_dpo.py inspect
.venv-m6/bin/python scripts/m6_dpo.py train
.venv-m6/bin/python scripts/m6_dpo.py verify
```

小样本配置固定20对、5次更新，学习者已完成并重载通过；6题前后对照在results/m6_dpo_smoke_01/development_timeline.md，未显示明确整体改善。更早开发检查的更新0次与实际训练5次分开记录；该入口不支持中断checkpoint恢复。

第三步仍从原M3 SFT采样，保留原40条，学习者依次运行：

```bash
.venv-m6/bin/python scripts/m6.py collect --split train --limit 180 --resume
.venv-m6/bin/python scripts/m6.py collect --split dev --limit 30 --resume
.venv-m6/bin/python scripts/m6.py review-pack --split train
.venv-m6/bin/python scripts/m6.py review-pack --split dev
```

扩展采样与审核已完成；原20组决策保留，开发不改写chosen凑量。冻结实际179组train、6组dev，训练拒收1、开发拒收24。主体为纠错，风格偏好仅2组较弱配对；开发类别覆盖不足，固定1轮终点，不做多轮试参。

第四步先inspect，再由学习者依次启动两条训练并在新进程重载：

```bash
.venv-m6/bin/python scripts/m6_train.py inspect
.venv-m6/bin/python scripts/m6_train.py train --method chosen_sft
.venv-m6/bin/python scripts/m6_train.py verify --method chosen_sft
.venv-m6/bin/python scripts/m6_train.py train --method dpo
.venv-m6/bin/python scripts/m6_train.py verify --method dpo
```

两条都从原M3 SFT开始，各1轮约45步，训练前及15/30/45步生成6条固定开发回答。相同步数不代表等算力。每条结果在results/m6_formal_01_<method>/，完整Adapter在outputs/m6_formal_01_<method>/adapter/policy/。入口不支持中断checkpoint恢复；不覆盖已完成run。正式DPO已由学习者完成并重载，chosen-only SFT待运行；新最终测试仍封存。训练记录见[验证记录](docs/M6_验证记录.md)，回答变化见[开发观察](results/m6_formal_01_dpo/development_timeline.md)。

本机采用 PyTorch MPS、Transformers 和 PEFT 的普通 LoRA，不是 4-bit QLoRA。`.venv` 使用 Python 3.12；完整实测依赖在 `requirements-lock.txt`。

## 开源复用

| 组件 | 来源 | 项目工作 |
|---|---|---|
| 基础模型、分词与聊天模板 | Qwen/Qwen2.5-1.5B-Instruct | 模型版本固定、角色提示词和本地加载 |
| 推理与训练循环 | Hugging Face Transformers | 数据接入、训练配置、运行记录 |
| LoRA 实现与权重管理 | Hugging Face PEFT | 适配模块选择、参数变化和重载验证 |
| 聊天界面与事件队列 | Gradio Blocks / Chatbot / State | 推理函数连接、会话隔离、模式切换、裁剪与错误恢复 |
| 教学示例与验证流程 | 本项目，AI 辅助开发 | 标签检查、流程验收、入门教程 |

参考官方模型卡、Trainer、PEFT 文档实现；未声称 fork 了完整 AI Waifu 应用，也未声称从零实现训练算法。Qwen 模型卡标注 Apache-2.0；各开源库许可证以对应发行版本为准。

## 结果与文件

- [M5评测报告](results/m5_learning_01/report.md)：240条回答、120对AI评分，逐对理由及AI来源均保存。重建当前报告运行`.venv/bin/python scripts/m5_ai_report.py`，无须再次生成或逐题打分。
- `models/`：基础模型与固定版本记录，只保存在本机。
- `outputs/`：Adapter 与临时训练输出，只保存在本机。
- `results/m1/`：环境、原模型回答、标签检查、训练和重载记录。
- `.venv/`：项目独立环境。

大文件、缓存与秘钥已配置忽略。M1 示例不含真实用户资料；M4已接入聊天界面，保留当前会话上下文，没有跨会话长期记忆或工具调用。

12 条示例和 8 次参数更新只用于 M1 流程验证，不能证明质量提升。
# ai_waifu
