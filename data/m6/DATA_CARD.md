# M6场景计划与偏好数据来源

版本m6-preference-v1；评分m6-three-axis-v1。学习者已生成train360/360、dev60/60条候选，失败0。210题全部由助手审核，现行快照pairs/38b37f1586fd7306/含train179组、dev6组；训练chosen原选26条、AI修订153条，开发无修订。训练拒收1题歧义问题，开发拒收24题；旧20组及原候选保留。学习者已完成旧20对DPO的5次更新与重载，以及正式179对DPO的45次更新与重载；chosen-only SFT和最终对照尚未完成。数据冻结记录为FORMAL_FREEZE.json。

| 集合 | 场景数 | 每类数 | 当前用途 |
|---|---:|---:|---|
| train | 180 | 30 | 用当前SFT采样两条回复，随后由助手审核偏好 |
| dev | 30 | 5 | 另采两条回复供偏好开发检查，不能入训 |
| test | 60 | 10 | 新最终测试封存；collect命令不接受test |

六类为daily、comfort、study、technical、stability、relationship；270条输入均由当前对话Codex原创设计，不是私人聊天、Galgame台词、当地服务信息或模型生成记录。少量前文assistant也是场景设计者编写的固定上下文，不是本次采样结果。作者未独立手写全部场景，所有模型候选必须另有实际生成记录。

`scripts/create_m6_scenarios.py`是来源，`scenarios.jsonl`是其序列化结果。先固定场景划分，再采样，不复制M5旧题到偏好数据。prepare检查旧数据及跨集合最后问题的相同文本和字符三元组近重复，阈值0.6；它没有做嵌入语义查重，不能证明全量语义隔离，主题和行为类型也可能相近。完整前文由family/split与消息格式契约控制；当前全部family_id独立，人工式语义划分仍有局限。

候选来自Qwen2.5-1.5B-Instruct＋m3_learning_01 SFT LoRA，使用M2冻结short提示词。每题相同输入采样两次，temperature=0.8、top_p=0.9、top_k=50、最多128新token；种子按场景与候选编号固定。每题都使用计划前文，不借用其他候选或其他题目的生成回答。expectation是审核重点，不进入模型输入。

原始候选写入results/m6_preparation/candidates.jsonl，逐条保存回答、seed、前文、结束原因、错误及来源条件哈希。随机左右阅读包放在该目录review/；私有对应关系在outputs/m6_preparation/review_keys/。失败或同文候选标不可直接配对，保留原始数据，不当作有质量的chosen。

助手按三个独立维度审核：口吻voice、情境与人设persona_context、有用性usefulness。优先检查事实、能力边界和用户要求，再比较自然风格；重要配对交换左右复核。技术内容需要核实；两边都差则reject，或明确记录AI修订来源后提供合格chosen，不自动把较不差者算好答案。接受的chosen人设/有用性至少4分；正式文体优先，不强求口癖。

实际审核记录已由助手写入review_decisions.jsonl。source与pack/candidates哈希绑定具体候选。export-pairs校验后按内容和审核文件哈希生成独立pairs/<版本>/train.jsonl、dev.jsonl和manifest.json；第一批快照train20组、dev空，现行快照train179组、dev6组，全部210题已审核。原始候选不覆盖。不同快照不能静默拼在一起。偏好全部AI提供，报告称AI反馈DPO，不称收集了人类反馈。

全量决策与原候选副本、逐组原左右评分、接受/修订/拒收理由保存在results/m6_preparation/reviews/expansion_*，原20组逐字保留。现行数据最大prompt＋回答210 token，未截断，末尾一个EOS；chosen较长71组、较短101组、等长7组。

训练五类各30、stability29；开发只有comfort2、study3、stability1，技术/日常/关系无合格原候选偏好，覆盖不足。正式路线固定1轮最终checkpoint，不用这6组不断试参；6条六类生成探针与保留的开发偏好不重叠，但已作开发观察，非未见测试。新最终60题继续封存。

主体是176组任务/情境纠错、2组较弱风格偏好、1组呈现偏好，不能声称充分覆盖自然角色风格。助手参与场景设计、改写153条chosen及评分，左右交换是已知身份的同一上下文复核，非独立双盲。35个技术问题有最小执行结果或官方依据；正式质量结论仍待同起点chosen-only SFT与DPO以及新最终对照。

第一批候选、决策和审核回执另固定在results/m6_preparation/reviews/batch_01_*，可读报告为batch_01_report.md。助手参与场景设计、改写18条chosen并评分，存在同源偏差；左右交换只是同一上下文复核，非独立双盲。原左右评分顺序为voice/persona_context/usefulness；修订后的chosen单独评分，不能拿它冒充原候选分数。技术题已执行Python事实检查，保留代码结果与官方依据。

当前20对最大prompt＋回答为177 token，均低于320上限，只有末尾一个EOS。chosen比rejected长10组、短9组、等长1组；这不证明评分没有长度偏差。18条新示范的教学信号需与chosen-only SFT对照区分，不能把一切收益归因DPO。

180/30是计划目标，不靠重复问法或低质量chosen补齐。候选齐备、偏好快照导出、DPO训练通过和新测试质量改善是四种不同状态。原小样本5次更新与重载完成，正式DPO也已由学习者完成179对、1轮45次更新与新进程重载；6题开发观察未支持明确整体改善。chosen-only SFT及最终质量对照仍待运行，见docs/M6_第四步_同起点训练对照.md及docs/M6_验证记录.md。

正式扩展遵循PRD：AI修订仅用于训练区，并明确记录；开发区只接受合格原候选间的清晰偏好，否则拒收，不修订chosen来凑30组。低保留量是数据质量结果，需据此决定是否另建开发采样版本；不能把坏回答降低门槛标成好答案。6条开发生成探针已经看过，仅作开发/回归，不称未见最终测试。
