# 数据目录

M1 教学样本 `m1_smoke.jsonl` 有12条，只用于验证推理、LoRA训练和重载流程。它保留原实验内容，不承担当前傲娇角色的正式训练。

M2 当前正式入口是 `m2/`：

- `raw_cards.jsonl`：120个原创训练场景族，含可选前文与三种问法。
- `train.jsonl`：360条展开样本，包含60条多轮上下文；M3已实际使用两轮。
- `dev_scenarios.jsonl`：24个开发场景，参考答案仅供核对，不进入推理输入。
- `test_scenarios.jsonl`：60个最终测试场景；不用于选择Prompt，目前未生成回答。
- `frozen_manifest.json`：本轮已生成，固定数据、Prompt、评分和关键代码指纹。

数量、来源和监督范围见 [M2 数据卡](m2/DATA_CARD.md)。人工抽查见 [30条台词](../results/m2/review_samples.md)。

`m2_tsundere_v2/` 留作首次候选快照，不能用其中旧状态代替当前主流程状态。旧游戏比喻版已归档到 `history/m2_persona_v1/`；改写前的傲娇v2及18pass/12revise意见已归档到 `history/m2_tsundere_v2/`。当前v3已落实学习/技术台词修订。

M3训练与验证见 [M3记录](../docs/M3_验证记录.md)，没有改写冻结的M2数据。
