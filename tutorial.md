# AI Waifu：M3–M6 逐步学习

用途：用原创角色对话对 Qwen2.5-1.5B-Instruct 做 LoRA 微调，观察角色语气与实际回答质量。

源码基准：M3历史commit `5c457ce`，M3开发观察、M4/M5交付commit `a502b7f`；M6源码与实测记录随本阶段提交保存。原始`configs/m3.json`和已有训练结果保留，正式实验由学习者亲手启动。

前置基础：能看懂Python函数调用、字典取值；已经了解loss、labels和LoRA。当前覆盖M3、M4、M5及M6候选、小样本DPO、全量审核与同起点对照入口。M6的420条候选与210题审核完成，现行179组train/6组dev已冻结；旧小样本5次更新及重载完成。正式DPO已由学习者完成45次更新与新进程重载，开发观察未支持明确整体改善；chosen-only SFT、最终评测与M7尚未完成。

## 01：训练配置（已讲解）

读完能解释：为什么 360 条数据训练两轮、一次处理一条、累计四次梯度再更新，会产生 180 次参数更新。

阅读 `configs/m3.json` 的 `num_train_epochs`、`batch_size`、`gradient_accumulation_steps`、`learning_rate`。配置由 `scripts/m3.py:preflight` 读取，传给 `train`，再传入 `TrainingArguments`。配置本身不会启动训练。

## 02：从基础模型走到训练（已展开）

目标：能指出哪里加载基础模型、哪里插入 LoRA、哪里真正开始参数更新。

主链：`train` → 校验配置与冻结数据 → `load_base` → `datasets` → `get_peft_model` → `TrainingArguments` / `Trainer` → 训练前 `evaluate` → `trainer.train` → 训练后检查与保存。

同一 run 已完成时，入口会校验并返回已有记录，不重新执行训练主链。以下讲的是首次训练的分支。

### 第一步：加载基础模型

必读 `scripts/m3.py:load_base`：

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, dtype=dtype,
    attn_implementation='eager', low_cpu_mem_usage=True
).to(device)
```

`MODEL` 是本地基础模型目录。Tokenizer 负责文字与 token 编号的转换；model 是执行预测的模型。加载后，磁盘中的权重进入运行所需的内存。`.to(device)` 将模型放到选定的计算设备；本机训练使用 MPS。

这一步没有传入 M3 Adapter，也没有用数据更新参数。它返回 `model, tokenizer, device`，供后续编码和训练使用。

### 第二步：插入 LoRA

必读 `scripts/m3.py:train` 中的 `get_peft_model`：

```python
model = get_peft_model(model, LoraConfig(
    r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
    lora_dropout=cfg['lora_dropout'],
    target_modules=cfg['target_modules'], bias='none',
    task_type='CAUSAL_LM', revision=read_json(MODEL_MANIFEST)['revision']
))
```

`cfg` 是配置字典，`cfg['lora_rank']` 读取我们已经看过的值 8。`LoraConfig` 描述怎样设置 LoRA；`get_peft_model` 执行插入并管理可训练参数。这里先创建新的 LoRA 参数，还没有学习凛音的数据。

当前 `target_modules` 是 `q_proj` 和 `v_proj`，属于注意力模块中的矩阵。现在只需记住：我们在模型的指定位置加入小矩阵，训练时更新这些小矩阵，保留基础模型参数。

下一段使用 `param.requires_grad` 找出允许计算梯度、供优化器更新的参数，并确认其名称都包含 `lora_`。具体注意力公式暂缓。

### 第三步：配置并启动 Trainer

必读 `scripts/m3.py:train` 中的 `TrainingArguments`、`Trainer` 和 `trainer.train`。下面只摘出本课需要的参数，完整代码还有其他设置：

```python
args = TrainingArguments(
    output_dir=str(trainer_dir),
    num_train_epochs=cfg['num_train_epochs'],
    learning_rate=cfg['learning_rate'],
    per_device_train_batch_size=cfg['batch_size'],
    gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
    # 其他参数见原文件
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=train_data, eval_dataset=dev_data,
    data_collator=default_data_collator,
    processing_class=tokenizer, callbacks=[Progress()]
)
result = trainer.train(resume_from_checkpoint=checkpoint)
```

`TrainingArguments` 提供训练规则；`Trainer(...)` 把模型、规则和数据组装起来；`trainer.train(...)` 才运行训练循环。循环读取 batch、执行预测、计算 loss、反向传播，按梯度累积设置更新 LoRA 参数。

创建 Trainer 后，脚本先调用一次 `trainer.evaluate`，计算训练前开发 loss；它不更新参数。首次训练的 `checkpoint` 是 `None`，表示没有从中断的训练状态恢复。训练完成后还会评估并保存，这部分留到后续课程。

### 串起来看

对于一条“今天很累”的角色示范对话：先加载 Qwen，准备数据的 token 与 labels，给 Qwen 加入新的 LoRA 参数，再让 Trainer 用示范回复计算 loss 并更新 LoRA。已有的 M1 Adapter 不参与这轮训练。

本课只要求复述：加载 model、调用 `get_peft_model`、创建 `Trainer`、调用 `trainer.train`，分别做什么，哪一步真正开始根据数据更新参数？无需运行项目。

选读验证证据：`results/m3_tsundere_v3_r8_e2/training.json` 中的 `trainable_parameters`、`changed_lora_tensors` 和 `global_step`。这些是此前代执行的结果，不是学习者本课刚完成的操作。

## 03：亲手准备并启动独立实验（已展开）

目标：使用新的实验名保留已有结果，亲手启动训练并解释第一段日志。

主链：选择新配置 → `main/preflight/paths` → 编码数据 → 训练前评估和生成 → `trainer.train` → 每轮评估触发 `Progress.on_evaluate` → `DevelopmentObserver.sample` → 继续训练 → 保存结果。

### 1：选独立配置，检查输入

阅读 `configs/m3_learning.json`。`run_id` 是 `m3_learning_01`，产物写入独立目录。训练数据和主要训练参数与已有实验相同；新增 `development_observation` 指定观察问题和生成长度。

终端执行：

```bash
cd "/Users/jakiimo/Desktop/AI Waifu"
.venv/bin/python scripts/m3.py inspect-data --config configs/m3_learning.json
```

`--config` 让本次命令读取学习配置；没有这个参数会读取原配置并复用已完成的旧实验。检查应显示360条训练、24条开发、60条带前文；打印出的监督文字仍是数据里的示范回复，不是模型生成。

### 2：亲手开始训练

```bash
.venv/bin/python scripts/m3.py train --config configs/m3_learning.json
```

程序重新加载原始基础模型，插入新的LoRA，不接着训练之前的Adapter。训练前先计算24条开发参考答案的loss，再生成6条实际回复；第1轮结束（90次更新）、第2轮结束（180次更新）重复相同的观察。

终端会打印各阶段回答。累计对照文档将在启动后生成于 `results/m3_learning_01/development_timeline.md`，原始记录为同目录的 `development_timeline.json`。不要把尚未生成的文件当作本轮完成证据。

### 3：区分开发loss和开发回答

`trainer.evaluate` 使用24条开发参考答案计算预测误差，不更新参数。`DevelopmentObserver.sample` 则只把6个场景的提示词和对话前文交给 `generate`，不提供最后的参考答案，让模型真正生成回复。

6题覆盖日常、安慰、教学、技术解释、倾听要求和上下文记忆。每个阶段保持同样的输入、short Prompt、贪心解码、128 token输出上限。128是本次推理生成上限，与训练的max_length=256不是同一个设置；显示 `length` 时，回答可能被截断。

观察通过 `Progress.on_evaluate` 挂到每轮评估结束。生成过程不反向传播；结束后恢复模型各模块的训练状态和随机数状态，然后训练继续。它会额外花时间生成回答。

阅读回答时分别判断：角色语气是否自然；技术是否准确（`labels=-100`意味着该位置不参与loss，不是输出负一百）；是否遵守“先听我吐槽”的要求；是否只根据前文记住“小猫叫薄荷”，没有编造其他信息。

loss下降与回答改善分别判断。不要因6题里某一句变好就宣称整体质量提升。最终Adapter按24条开发loss选checkpoint，可能是第1轮；对照文档会标记选中的更新步数。最终测试集仍留到M5。

### 4：等待与查看

上一轮本机实际训练约13分钟；本轮额外生成开发回答，时长会增加且受机器状态影响。第一次观察完成后，可以打开对照文档，每轮结束重新打开查看新增回答。另一终端查看：

```bash
.venv/bin/python scripts/m3.py status --config configs/m3_learning.json
```

同一run完成后不会重复训练。若中断且已有完整checkpoint，可使用同一配置执行 `train --resume`；若首个checkpoint保存前中断，不能恢复，保留记录并建立新run。恢复时开发观察会移除超过最后已保存checkpoint的记录。

本课先完成训练和观察，暂不执行verify或dev-compare。下一课再解释磁盘保存与重载。

本课复述任务：为什么开发观察不提供最后的参考答案？为什么loss下降仍需要阅读实际回复？运行后记录同一问题在训练前、第一轮、第二轮的变化，也可以如实记录“没有变好”。

选读验证证据：`tests/test_m3_observation.py` 使用临时CPU微型随机模型验证每轮回调，不是正式Qwen模型训练或效果证据。正式效果只有亲手运行后产生的记录才能说明。

后续：样本编码与监督区域、日志和开发 loss、保存与独立进程重载、实际回答比较。保存和恢复分支将在相应课程结合真实代码展开。

## 04：把训练结果接成聊天 Demo（已展开）

M3学习者实验已完成训练、独立进程重载和18条开发对照。现在使用其Adapter做推理，不再训练。

本课读完能回答：网页输入怎样走到模型实际回复，下一轮怎样携带历史，原模型模式为何不需要加载第二份Qwen。

主链：`demo.py:main` → `ChatEngine`核对产物 → 首次输入触发`ensure_loaded` → `reply` → `prepare_messages` → `_generate` → 解码并保存当前会话历史 → Gradio显示。

必读顺序：

1. `configs/m4.json`：指定已完成的训练配置、聊天/比较Prompt和输入输出上限。
2. `demo.py:ensure_loaded`：读取基础模型和磁盘中的A/B，检查参数冻结，返回可推理的模型。
3. `demo.py:reply/prepare_messages`：文字和历史转成消息列表，必要时裁掉最早完整轮次；当前问题过长则停止并提示。
4. `demo.py:_generate`：根据mode禁用或启用Adapter，`inference_mode`下生成，切掉输入token后解码。
5. `demo.py:build_app`：Gradio事件绑定`send_chat`；每个会话由`gr.State`保存历史，`reset_chat`清空，`compare_models`使用独立同题输入。

第一次输入会加载权重，以后复用模型。生成失败时保留原历史并恢复按钮；模式切换和新会话重置历史；同题比较顺序执行且与聊天历史独立。

具体亲手命令、源码摘录与网页体验步骤见[本课教程](docs/M4_入门教程.md)。按`--check` → `--message` → 网页启动顺序执行。回顾任务：用自己的话复述“新会话”究竟清掉了什么，以及模型如何知道上轮小猫的名字。

选读证据：`tests/test_m4_demo.py`和`docs/M4_验证记录.md`。M5最终盲评与长期记忆不属于本课。

用户已直接反馈M4教程操作完成，反馈记录见results/m4/learner_acceptance.json。此处只记录操作，不推断模型质量合格。

## 05：四组对照与匿名人工评测（已展开）

进度更新：原人工路线保留作课程说明；用户后续将评分委托给助手，240条正式回答及120对AI评分已完成，报告见results/m5_learning_01/。不冒充人工盲评，原文“0/240”指当时准备阶段。

读完能回答：如何区分LoRA的额外贡献与详细Prompt的帮助，如何保存同题条件，如何把人工偏好汇总为有明确分母的结果。

主链：`configs/m5.json` → `scripts/m5.py:preflight/prepare/build_tasks` → `generate/generate_records` → `build_pairs/blind` → `scripts/m5_review.py:save_rating` → `parse_ratings/summarize/report`。

1. `prepare`核对60题冻结测试、四组条件、评分版本和Adapter，用Tokenizer统计240个输入长度，保存manifest与源码快照；没有模型回答。
2. `generate`复用M4引擎，先核对M3预测探针，再给每题四组相同预置历史。逐条保存完整模型输入、回答和错误；中断只继续未保存任务。
3. `blind`将B/D与A/C配对，左右平衡随机、身份隐藏，共120对；每条回答只出现一次。失败对保留但不伪造人工评分。
4. `m5_review.py`只读取匿名答卷，评分由学习者本人选择；每次保存先校验，再原子替换CSV。页面刷新重新读取文件，不丢已保存进度。
5. `report`按私有映射计算胜平负、四组人工摘要、类别结果和错误，保存评分快照。缺评分只能出partial；有效样本不足只能出诊断报告，不标为M5完成。

具体亲手命令与评分规则见[M5教程](docs/M5_入门教程.md)。准备包与代码已检查；当前正式回答0/240、人工评分未开始。选读证据：tests/test_m5_evaluation.py、docs/M5_验证记录.md，测试仅用临时QA数据，不是正式效果结论。

回顾任务：解释D/B为何能隔离Adapter影响，而C/B不能；拿一个假设的胜平负数量，计算含平局与排除平局两种胜率，并写清分母。评分完成后根据真实结果讨论技术取舍，不为微调预设胜利。

## 06：为DPO准备同题候选（已展开）

读完能回答：为什么要两条答案，为什么生成候选还不是训练，谁决定chosen/rejected。

主链：终端collect → scripts/m6.py:main/prepare → preference_data.py:build_candidate_tasks → CandidateEngine.ensure_loaded/generate → generate_candidates保存 → review_pack → 助手实际审核 → export_pairs。

必读顺序：configs/m6.json（同一short、两条采样、独立seed）→ build_candidate_tasks（messages不含expectation）→ CandidateEngine.ensure_loaded（加载旧SFT并冻结）→ generate（实际推理）→ generate_candidates（中断与缓存）→ validate_decision/export_pairs（审核门槛与来源绑定）。每步的输入、下一调用、结果与失败恢复见[M6入门教程](docs/M6_入门教程.md)。

本课亲手操作只到前20题的40条候选与审核包，不要求学习者打标。双方都差不能自动选一个当好答案；由助手核查或记录修订。拒绝使用test作候选集，生成失败和同文回复保留，不悄悄补成功记录。

选读tests/test_m6_preference.py：临时CPU微型TRL实验不是正式Qwen DPO。当前40条候选已完成，20组实际偏好已审核导出，继续第07课；PPO暂缓。

回顾任务：复述这40条回复由哪个模型生成，collect是否更新A/B，以及chosen/rejected与M3单个示例回答的区别。

## 07：从偏好对到DPO参数更新（已展开，学习者已训练并重载）

目标：解释DPO同时比较两条已知答案的概率、相对冻结SFT的偏好变化，以及为什么只有policy LoRA更新。实际操作见[M6第二步教程](docs/M6_第二步_DPO小样本教程.md)。

输入是20组固定prompt/chosen/rejected，2组直接选择原候选、18组chosen由助手修订；输出是更新5次的完整policy LoRA、训练日志、6条开发题前后对照和重载记录。全部偏好为AI反馈，修订来源不得隐去。

主链：inspect核对真实数据 → train/preflight验证来源 → encoded_rows/encode_preference确定回答与EOS边界 → make_model复制policy/reference → make_trainer建立TRL训练器 → observe生成6题训练前回答 → 固定参考概率 → trainer.train计算loss/反向/更新 → observe生成训练后回答 → 保存policy → 新进程verify。

必读顺序：configs/m6_dpo_smoke.json（20对、batch1、累积4、5次更新）→ preflight（审核与候选来源绑定）→ encode_preference（只对最后回答计目标概率）→ make_model/copy_reference_from_policy（原SFT精确复制、reference冻结）→ trainer_class/make_trainer（TRL DPO与参考上下文）→ train/observe（训练与开发生成分开）→ verify（重载探针）。每个环节的输入、结果与错误保留见第二步教程。

概率通过teacher forcing计算：喂入保存的真实答案，并读出每个目标token的概率，不在训练中重新生成chosen。目标提升policy的chosen/rejected差值相对reference的变化；起点margin0、loss约0.693不表示两答案等概率。DPO日志的reward是隐式概率比，不是额外奖励模型。

开发检查：17项相关测试通过；真实Qwen/MPS上两对前向与最长样本一次反向通过，optimizer更新0次。学习者随后完成实际5次更新，112个policy LoRA张量改变，reference不变，新进程重载logit探针差0。6题开发观察未显示明确整体改善。该入口不支持中断checkpoint恢复；失败保存记录后另开run，不静默重做。

回顾：指出哪一步第一次改A/B；解释为何不能关闭LoRA作reference；说明rejected在loss中的作用；读6题前后回答并列出改善与退化，不能用loss下降代替最终评测。

chosen-only SFT、完整数据训练、新最终测试与M7 PPO仍待后续步骤实施。

## 08：扩展并审核训练/开发偏好（已展开，采样与审核完成）

目标：区分场景数、候选数与审核后偏好数；理解续采如何保留已生成结果，以及开发区为什么不能靠改写凑量。完整操作与原理见[M6第三步](docs/M6_第三步_扩展偏好数据.md)。

输入仍为原M3 SFT＋同一short＋预先划分场景，不用小样本DPO模型。180题训练目标需360条候选，已有40条，所以train --limit180 --resume新增320条；30题开发另生成60条。最终test60题不采样。

主链与必读：collect → prepare（冻结条件）→ build_candidate_tasks（split和前N题）→ generate_candidates（known key验证、跳过已保存项）→ CandidateEngine.generate（参数冻结采样）→ 逐条保存 → review_pack（新哈希包）→ 助手实际审核 → export_pairs（保留旧快照、导出新快照）。恢复与错误保留说明在第三步教程中；选读原恢复/泄漏测试，暂缓正式训练器和PPO。

新train包会重复展示前20题，原审核按case_id保留，不重复决策。训练chosen修订记录AI来源，开发两边都差则拒收，不改写开发chosen凑30组。记录纠错对/风格对、类别与长度分布；候选数420不等于可用偏好210。数据不足时如实说明，再定扩展版本。

回顾：解释新增320而不是360的原因；描述中断后已保存的候选和生成错误分别怎样处理；说明为什么chosen-only SFT与正式DPO都从原M3 SFT开始。实际210题审核后179组train/6组dev，训练拒收1、开发拒收24，原20组保留。新快照与冻结记录见data/m6/FORMAL_FREEZE.json，继续09。

## 09：同起点chosen-only SFT与DPO（DPO已完成，SFT对照待运行）

目标：区分新示范的收益与偏好目标的收益；亲手完成两条独立训练/重载，并观察第0、15、30、45步的开发变化。操作见[M6第四步](docs/M6_第四步_同起点训练对照.md)。

输入179组训练偏好：26条chosen原选、153条AI修订；6组开发偏好均原选，覆盖有限，不做试参。6条生成探针在开发区，与这6组偏好不重叠。两条路线从原M3 SFT复制，固定1轮最后checkpoint，不从小样本DPO继续。预期45步，DPO两回答与参考计算，不是等算力。

主链与必读：scripts/m6_train.py:main → preflight/validate_rows（冻结SHA、实际审核、split与候选绑定）→ sft_rows（只监督chosen＋EOS）或make_trainer的DPO分支（两答案相对原SFT）→ m6_dpo.make_model（精确复制policy/reference）→ train/observe(step0) → Trainer.train → Progress.on_step_end（15步生成并恢复状态）→ 保存完整policy → 新进程verify。

SFT保留prompt的上下文但labels=-100，rejected不进入其输入；DPO仍用冻结SFT参考。评价loss定义不同，不按数值大小判哪条路线更好。选择固定一轮终点，最终60题尚未生成；生成探针已看过，只叫开发回归。

选读tests/test_m6_formal.py：覆盖开发修订拒绝、偏好来源篡改拒绝、全量SFT目标mask、临时真实Trainer的末组梯度累积、参考/基础不更新及policy保存重载。暂缓最终评分与PPO。真实Qwen反向开发检查不更新参数；实际正式训练由学习者启动。

失败分支：条件变更训练前停止；已有未完成run保留，不自动checkpoint恢复；训练/生成失败写progress；参考变化、非有限值或重载不一致报错。成功run重复train核对产物后退出。

回顾：说明SFT中的rejected用途；计算179/累积4为何45步；对比同6题四个时点，列出改善、未变和退化。正式DPO已完成，记录见results/m6_formal_01_dpo/：参数更新和重载成功，但开发观察没有明确整体改善。完成chosen-only SFT及重载后，再展开四模型新60题对照与质量报告。
