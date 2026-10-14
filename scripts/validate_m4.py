"""开发者检查M4真实推理链路，不训练、不启动用户网页、不执行最终测试。"""
from pathlib import Path
import importlib.metadata
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import demo
from prepare_data import dump_json, digest, read_json
from scripts.m1 import now, versions
from scripts.m3 import logits_for


def main():
    engine = demo.ChatEngine()
    adapter_weight = engine.adapter / 'adapter_model.safetensors'
    before_hash = digest(adapter_weight)
    first = engine.reply('我给小猫取名松饼，今天刚接它回家。')
    second = engine.reply('刚才提到的小猫叫什么？', first['history'])
    comparison = engine.compare('今天把书桌整理好了，想听一句鼓励。')
    frozen_after_demo_calls = not any(p.requires_grad for p in engine.model.parameters())
    import torch
    from safetensors.torch import load_file
    record = read_json(engine.adapter.parent / 'run_manifest.json')
    refs = load_file(str(engine.adapter.parent / 'probe_reference.safetensors'))
    with engine.model.disable_adapter():
        base_logits = logits_for(engine.model, engine.tokenizer, record['verification_messages'], engine.device)
    # 此处直接调用PEFT做外部数值探针，同样恢复其退出上下文后的参数标志。
    engine.model.requires_grad_(False)
    adapted_logits = logits_for(engine.model, engine.tokenizer, record['verification_messages'], engine.device)
    checks = {
        'all_parameters_frozen': frozen_after_demo_calls,
        'adapter_file_unchanged': digest(adapter_weight) == before_hash,
        'history_passed_to_second_turn': second['messages'][1:-1] == first['history'],
        'comparison_is_independent': len(comparison['messages']) == 2,
        'base_logits_match_m3': bool(torch.allclose(base_logits, refs['base_logits'], rtol=1e-3, atol=1e-2)),
        'adapter_logits_match_m3': bool(torch.allclose(adapted_logits, refs['adapted_logits'], rtol=1e-3, atol=1e-2)),
    }
    report = {'recorded_at_utc': now(), 'purpose': 'developer integration smoke, not learner acceptance or M5 evaluation',
              'run_id': engine.train_cfg['run_id'], 'device': engine.device,
              'weight_dtype': str(next(engine.model.parameters()).dtype), 'packages': versions(),
              'gradio': importlib.metadata.version('gradio'), 'adapter_sha256': before_hash,
              'source_sha256': digest(ROOT / 'demo.py'), 'config_sha256': digest(demo.DEFAULT_CONFIG),
              'checks': checks, 'first_turn': first, 'second_turn': second, 'comparison': comparison,
              'final_test_executed': False, 'training_executed': False}
    dump_json(ROOT / 'results/m4/inference_validation.json', report)
    print('M4开发检查：' + str(checks), flush=True)
    if not all(checks.values()):
        raise RuntimeError('M4真实推理接入检查未通过。')


if __name__ == '__main__':
    main()
