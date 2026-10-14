"""观察必须隔离开发参考答案、恢复训练状态，并真实挂到每轮评估。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import scripts.m3 as m3


class DevelopmentObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m3.read_json(m3.ROOT / 'configs/m3_learning.json')
        self.m2 = m3.read_json(m3.M2_CONFIG)
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self):
        from types import SimpleNamespace
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
        model.config = SimpleNamespace(use_cache=False)
        return model

    def generated(self, model, tokenizer, messages, device, settings):
        model.eval()
        return {'messages': messages, 'answer': '测试用生成结果', 'input_tokens': 3,
                'output_tokens': 3, 'finish_reason': 'eos', 'seconds': 0.0}

    def test_rejects_test_cases_duplicates_and_changed_prompt(self):
        for settings in [
            {'case_ids': ['test-01']}, {'case_ids': ['dev-01', 'dev-01']},
            {'prompt_name': 'strong_examples'}, {'max_new_tokens': 0},
        ]:
            cfg = deepcopy(self.cfg)
            cfg['development_observation'].update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                m3.validate_config(cfg, self.m2)

    def test_restores_mixed_training_modes_cache_rng_and_weights_even_on_error(self):
        observer = m3.DevelopmentObserver(self.cfg, self.m2, None, 'cpu', self.results)
        model = self.model().train()
        model[1].eval()  # 不能只保存最外层的training标志。
        modes = [module.training for module in model.modules()]
        weights = [parameter.detach().clone() for parameter in model.parameters()]
        cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()

        def broken(*args):
            model.eval()
            model.config.use_cache = True
            torch.rand(3); random.random(); np.random.rand(3)
            raise RuntimeError('模拟生成中断')

        with patch.object(m3, 'generate', side_effect=broken), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '生成中断'):
                observer.sample(model, 0, 0.0, 4.0)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertFalse(model.config.use_cache)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
        for before, after in zip(weights, model.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertFalse(observer.path.exists())

    def test_same_messages_without_reference_answer_duplicate_step_and_resume_pruning(self):
        observer = m3.DevelopmentObserver(self.cfg, self.m2, None, 'cpu', self.results)
        model = self.model()
        with patch.object(m3, 'generate', side_effect=self.generated) as generate, redirect_stdout(io.StringIO()):
            observer.sample(model, 0, 0.0, 4.0)
            observer.sample(model, 90, 1.0, 3.5)
            observer.sample(model, 180, 2.0, 3.4)
            observer.sample(model, 180, 2.0, 3.4)
        self.assertEqual(generate.call_count, 18)
        cases = {c['id']: c for c in m3.read_jsonl(m3.DEV)}
        for row in observer.report['snapshots'][0]['records']:
            self.assertEqual(row['messages'][1:], cases[row['case_id']]['messages'])
            self.assertNotIn('reference_answer', row)
        for first, second in zip(observer.report['snapshots'][0]['records'], observer.report['snapshots'][1]['records']):
            self.assertEqual(first['messages'], second['messages'])
        observer.finalize('checkpoint-90')
        self.assertEqual(m3.read_json(observer.path)['selected_checkpoint_step'], 90)
        resumed = m3.DevelopmentObserver(self.cfg, self.m2, None, 'cpu', self.results, resume_step=90)
        self.assertEqual([s['step'] for s in resumed.report['snapshots']], [0, 90])
        self.assertNotIn('selected_checkpoint_step', resumed.report)

    def test_actual_trainer_epoch_callback_in_temporary_cpu_toy_run(self):
        # 随机初始化的微型模型，只验证接线；不是Qwen基础权重训练或效果证据。
        from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

        cfg = deepcopy(self.cfg)
        cfg.update(run_id='m3_toy_test', gradient_accumulation_steps=1, lora_rank=2, lora_alpha=4)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        ))

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return torch.tensor([[1, 3, 4]])

            def save_pretrained(self, folder):
                (Path(folder) / 'toy_tokenizer.json').write_text('{}')

        rows = [{'input_ids': [1, 3, 4, 5, 6, 2], 'attention_mask': [1]*6,
                 'labels': [-100, -100, -100, 5, 6, 2]} for _ in range(4)]
        tokenizer = Tokenizer()
        cfg_path = self.results / 'toy_config.json'
        m3.dump_json(cfg_path, cfg)

        def toy_generate(model, tokenizer, messages, device, settings):
            model.eval()
            with torch.inference_mode():
                token = model(input_ids=torch.tensor([[1, 3, 4]])).logits[0, -1].argmax().item()
            return {'messages': messages, 'answer': f'toy token {token}', 'input_tokens': 3,
                    'output_tokens': 1, 'finish_reason': 'eos', 'seconds': 0.0}

        frozen = m3.read_json(m3.FROZEN)
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch.object(m3, 'ROOT', self.results), patch.object(m3, 'CONFIG', cfg_path), \
                 patch.object(m3, 'preflight', return_value=(cfg, self.m2, frozen)), \
                 patch.object(m3, 'protected_hashes', return_value={}), \
                 patch.object(m3, 'load_base', return_value=(model, tokenizer, 'cpu')), \
                 patch.object(m3, 'datasets', return_value=(rows, rows[:2], [])), \
                 patch.object(m3, 'generate', side_effect=toy_generate), \
                 patch('transformers.TrainingArguments', side_effect=lambda **kw: TrainingArguments(**kw, use_cpu=True)), \
                 redirect_stdout(io.StringIO()):
                record = m3.train()
        finally:
            torch.set_num_threads(original_threads)
        timeline = m3.read_json(self.results / 'results/m3_toy_test/development_timeline.json')
        self.assertEqual([s['step'] for s in timeline['snapshots']], [0, 4, 8])
        self.assertEqual([s['epoch'] for s in timeline['snapshots']], [0, 1, 2])
        self.assertEqual(len(timeline['snapshots'][0]['records']), 6)
        self.assertGreater(record['changed_lora_tensors'], 0)
        self.assertTrue(record['base_logits_unchanged_after_training'])
        self.assertIn(timeline['selected_checkpoint_step'], [4, 8])


if __name__ == '__main__':
    unittest.main()
