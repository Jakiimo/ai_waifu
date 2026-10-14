# M6第二步：亲手跑20对、5次更新的DPO

第一步你已生成40条真实SFT回答并创建审核包。我已审核并导出20组偏好。本课操作现已完成：学习者实际更新5次，112个policy LoRA参数张量改变，reference未变，新进程重载探针logit差为0。6题开发观察未显示明确整体改善，见[结果解读](../results/m6_dpo_smoke_01/学习结果解读.md)。继续[M6第三步](M6_第三步_扩展偏好数据.md)；以下保留本课操作说明，完整M6尚未完成。

## 1. 我已经处理了什么

40条原始回答全部保留。其中2组直接选出可用的原回答，18组两边存在事实、指令遵循、编造前文或训话问题，我明确修订了chosen；每组rejected仍来自你采样的真实SFT回答。不能把“比较不差”自动标为好答案。

所以这批数据是**AI反馈＋大量AI修订**，不是20组都从两条原回复中选赢家，也不是人类反馈。修订示例会带来新的教学信号，后面需要chosen-only SFT对照，才能判断改善是否需要DPO。交换左右的复核也是同一助手完成，不能当独立双盲实验。

可读审核报告：[第一批审核结果](../results/m6_preparation/reviews/batch_01_report.md)。实际训练文件是`data/m6/pairs/8c42c402efe59c25/train.jsonl`。你无需继续逐题标注。

## 2. 先看一条真正进入训练的数据

在项目终端执行：

```bash
cd "/Users/jakiimo/Desktop/AI Waifu"
.venv-m6/bin/python scripts/m6_dpo.py inspect
```

它只读取数据和Tokenizer，**不加载Qwen、不生成回答、不更新参数**。打印的问题是`range(2, 7, 2)`，chosen给出`2、4、6`，rejected漏掉`2`。打印的两条回答是已经保存的文字，不是这一刻模型新回答的。

看清三个部分：prompt是固定short角色提示词＋可能的历史＋新问题；chosen是希望更受偏好的完整回答；rejected是同一输入下的较差回答。本样本chosen为原SFT输出，其他18组会注明AI修订来源。Token统计包括各回答末尾一个EOS。

## 3. 这次模型到底在学习什么

SFT给模型一个示范，训练它提高示范答案的概率。DPO同时给同题的chosen与rejected，训练它相对起点更偏向chosen。

```text
                  同一份M3 SFT LoRA
                     复制两份
                 /              \
       policy：允许更新       reference：冻结
                 \              /
          同一个prompt＋chosen／rejected
                        ↓
          各自计算两条已知答案的对数概率
                        ↓
                比较相对偏好，得到loss
                        ↓
             反向传播，只更新policy的A/B
```

两份LoRA共用冻结的基础模型，省去第二份大权重。reference是“Qwen＋原M3 SFT LoRA”，不是关闭LoRA后的裸Qwen。参考参数只在开始前精确复制一次；之后保持不变。本次会提前计算参考概率并保存，训练时复用这些数值。

**这里没有让模型重新生成chosen。** 我们把已保存的回答接在prompt后面，模型根据前面的真实文字预测每个下一token，逐项读取这些token的概率。这叫teacher forcing（教师强制）。例如看到问题及答案开头`2`，再看模型给后续逗号、`4`等token多大概率。

将最后回答各token的log概率加起来，得到整段回答的log概率。只累计最后回答及EOS；问题、历史与padding不累计，但仍提供上下文。DPO的目标可以写成：

```text
policy差值 = logP_policy(chosen) - logP_policy(rejected)
reference差值 = logP_ref(chosen) - logP_ref(rejected)
margin = policy差值 - reference差值
loss = -log(sigmoid(beta × margin))
```

beta本次为0.1。训练希望margin上升，意思是**相对原SFT，更偏向chosen**。并不要求不同长度的chosen绝对概率必然超过rejected，也不能把概率变化直接当聊天质量分数。

刚开始policy与reference相同，所以margin约0、loss约`0.693147`。这不表示两条答案等概率，也不是训练失败。反向传播把loss对A/B的影响算成梯度；优化器才根据梯度更新参数。训练后的policy直接保存为完整新LoRA，不进行模型权重相减。

可选运行`.venv-m6/bin/python scripts/m6_loss_demo.py`：它只用虚构概率演示公式，不加载或训练Qwen。

## 4. 现在由你启动小样本训练

先关闭仍在运行的聊天Demo或其他模型进程。保持`.venv-m6`环境，执行：

```bash
.venv-m6/bin/python scripts/m6_dpo.py train
```

本次配置见`configs/m6_dpo_smoke.json`：20组偏好、学习率`1e-5`、每个小batch一组偏好、累积4组梯度后更新一次，共5次参数更新。这20组按训练器打乱后的顺序走一遍；一组偏好有两条回答，并不是仅训练chosen。

终端按以下顺序出现：

1. 加载现有Qwen和M3 SFT，建立policy/reference。
2. 打印6个新开发问题的**DPO前**回答，一类一题；不给模型审核重点或参考答案。
3. 计算并固定20组参考概率。此时尚未更新policy。
4. 出现“现在trainer.train才开始更新policy LoRA”，随后输出5步训练日志。
5. 打印同6题的**DPO后**回答，保存训练产物和对照文件。

这次观察训练前与第5次更新后两个时点。完整训练以后再扩展为中途多检查点观察。这6题不参加训练，也不使用封存的最终测试题；这里只观察，不凭它选最佳checkpoint。

加载日志中的“未加载M1 Adapter”来自基础模型加载函数；后面“已复制M3 SFT”才是这次加载的训练起点。

我已经完成17项相关测试，并在真实Qwen/MPS上核对参考概率、做最长样本一次反向传播；检查没有执行optimizer.step，参数未更新。你的这条`train`命令才会进行真实Qwen的5次参数更新。开发检查不保证完整训练必然成功，完整训练、保存、重载还需这次实际结果验证。

## 5. 日志怎么看

- `step`：参数更新次数，目标到5；不是采样回答数。
- `loss`：这批偏好目标的数值，最初接近0.693；小批次会波动，下降不等于聊天质量提升。
- `rewards/chosen`、`rewards/rejected`：DPO根据相对reference的log概率算出的隐式reward，不是额外奖励模型的打分。
- `rewards/margins`：上述chosen与rejected隐式reward的差；越正表示policy相对reference越偏向chosen。
- `rewards/accuracies`：当前批次中margin大于0的比例，不是技术问答正确率或开发集胜率。
- `grad_norm`：梯度大小。若出现NaN、Inf或报错，保存终端输出交给我定位。

仅5次更新可能看不到明显语言变化，或个别回答变好、变差。此次先确认真实训练链路能完成，效果判断留到数据扩展与对照实验。

## 6. 完成后确认能重新加载

训练进程退出后，另执行一条命令，得到新的Python进程：

```bash
.venv-m6/bin/python scripts/m6_dpo.py verify
```

它读取刚保存的Adapter，核对文件和来源，再比较新进程的logits与保存时的探针；不训练。成功后运行：

```bash
.venv-m6/bin/python scripts/m6_dpo.py status
```

主要结果：

| 文件 | 意义 |
|---|---|
| `results/m6_dpo_smoke_01/development_timeline.md` | 6题训练前后对照，最先读它 |
| `results/m6_dpo_smoke_01/training_log.json` | 逐步训练指标 |
| `results/m6_dpo_smoke_01/training.json` | 实际更新次数、参数检查、耗时、内存采样与来源 |
| `results/m6_dpo_smoke_01/reload_verification.json` | 新进程重载结果 |
| `outputs/m6_dpo_smoke_01/adapter/policy/` | 完整DPO后LoRA，应单独配原基础模型使用 |
| `outputs/m6_dpo_smoke_01/reference_probabilities.json` | 原SFT固定的参考概率 |

新Adapter已包含M3起点及本次更新，不能再叠加旧M3 Adapter。M4现在的默认模型仍按原配置运行，等学习验证完成再决定如何加入新模式。

该小样本入口暂不支持中断checkpoint恢复。若中断或失败，不删除旧记录反复跑；把错误给我，保留证据后用新的run配置重做。已经成功完成的run再次运行train会检查产物并退出，不重复训练。

## 7. 这课的源码路线

必读：`scripts/m6_dpo.py:main → preflight → encoded_rows/encode_preference → make_model → make_trainer → train → observe → verify`。

- `preflight`核对这20组来自实际审核，划分、原候选、修订与来源没有被改动。
- `encode_preference`定义文字和token边界；`make_trainer`核对TRL实际编码一致，避免截断或重复EOS。
- `make_model/copy_reference_from_policy`建立两份相同但独立的LoRA；Mac实测发现第二次加载会先舍入，因此先核对磁盘原参数，再精确复制。
- `trainer_class`使用TRL的DPOTrainer，仅补充冻结参考上下文的异常恢复及FP32损失比较。训练循环、DPO目标与优化器使用开源实现。
- `train`固定参考概率，再由Trainer反向传播、更新policy、记录日志；`observe`只生成开发回答并恢复训练状态。
- `verify`检查新进程重载，不能用“能生成一句话”代替数值核对。

选读`tests/test_m6_dpo.py`，暂缓深入PPO。读完应能解释：为什么两份Adapter起点相同？为什么参考不能一起训练？rejected是否进入loss？训练数据的18条修订是谁写的？5步日志是否足以证明质量改善？

下一步在你完成train与verify后展开：增加训练/开发偏好、chosen-only SFT对照、正式DPO与新最终测试。M7奖励模型与PPO随后进行。

参考：[TRL 0.24.0 DPO文档](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)、[DPO原论文](https://arxiv.org/abs/2305.18290)。
