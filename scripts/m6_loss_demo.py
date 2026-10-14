"""只演示DPO公式和梯度方向，不加载Qwen，不更新任何Adapter。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from preference_data import dpo_loss
    chosen = torch.tensor([-5.0], requires_grad=True)
    rejected = torch.tensor([-6.0], requires_grad=True)
    reference_chosen = torch.tensor([-5.0])
    reference_rejected = torch.tensor([-6.0])
    loss, margin = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1)
    loss.mean().backward()
    print('这是虚构标量的公式演示，不是Qwen训练，也没有模型回答。')
    print('训练起点与参考相同：margin=', round(margin.item(), 6), '；loss=', round(loss.item(), 6))
    print('chosen log概率的梯度：', round(chosen.grad.item(), 6))
    print('rejected log概率的梯度：', round(rejected.grad.item(), 6))
    print('梯度下降会推动相对偏好差增大；真实训练通过反向传播更新LoRA，不直接修改这两个标量。')


if __name__ == '__main__':
    main()
