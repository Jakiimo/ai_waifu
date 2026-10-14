# m3_learning_01 实验归档

本轮由学习者在终端实际启动并完成。训练360条实例×2轮、180次参数更新；选中checkpoint-180。

- `terminal_log.txt`：用户粘贴的原始终端记录，未经改写。开头的无--config dev-compare指向此前实验，不代表这轮的新对照。
- `development_timeline.md` / `.json`：本轮训练前、第一轮、第二轮，同样6个开发场景的实际回答。
- `training.json` / `training_log.json`：实际训练、参数更新、开发loss记录。
- `evidence_index.json`：本轮归档文件SHA-256及来源。基础模型只保留版本与哈希指向，不重复打包约2.88GiB的权重。
- 本轮Adapter和两轮checkpoint保存在 `outputs/m3_learning_01/`。
- 完整实验包：`outputs/archives/m3_learning_01_complete.tar.gz`，含本轮Adapter、checkpoint、原始LoRA与预测参考分数、日志、训练配置/源码快照、冻结数据和相关依赖文件。

本轮保存时尚无独立进程reload_verification或训练后dev-compare记录；已有的是训练进程内的保存检查和六题阶段观察。最终测试未执行。不把口吻变化或loss下降当作整体质量提升证据。

归档未修改训练配置、数据或模型权重。包内保留当时的本机路径和环境记录；不是跨机器开箱即用的部署包。
