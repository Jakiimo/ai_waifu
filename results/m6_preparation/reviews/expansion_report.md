# M6扩展审核与数据冻结

训练179组、开发6组；训练拒收1组（问题歧义），开发拒收24组。原20组决策与原始420条候选保留。

训练chosen直接选原候选26条、AI修订153条；开发6条全部原候选，未改写。多数数据承担任务、上下文和边界纠错，不声称已充分覆盖风格优化。

训练六类：daily/comfort/study/technical/relationship各30，stability29。开发仅comfort2、study3、stability1，覆盖不足；开发loss/margin不能代表全面效果，不据此做多轮参数搜索。

只有2组较弱风格偏好与1组呈现偏好。chosen较长71组、较短101组、等长7组；不把长度或口癖数量当评分规则。这是描述分布，不证明不存在偏差。

185组配对最大prompt＋回答210 token，均低于320，末尾一个EOS，未截断。35个技术问题已做最小代码检查或官方资料核对；相同种子跨设备不保证一致依据[PyTorch 2.8官方说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)，detach共享存储依据[官方接口](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html)。

审核、改写与左右交换复核均由同一对话Codex完成，已知答案身份，不是独立双盲；精确后端未知，全部AI反馈，人类偏好0条。

固定文件见data/m6/FORMAL_FREEZE.json；实际偏好快照为data/m6/pairs/38b37f1586fd7306/。下一步同起点chosen-only SFT与DPO各1轮，固定最终checkpoint，记录预算差异；正式训练与最终评测尚未执行。

[全部训练决策](expansion_train_report.md) · [全部开发决策](expansion_dev_report.md)
