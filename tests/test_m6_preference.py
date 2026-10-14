"""M6数据/采样契约与CPU微型TRL验证；不生成正式候选、不训练真实Qwen。"""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preference_data import (build_candidate_tasks, validate_scenarios, validate_decision,
                             encode_preference, dpo_loss, completion_log_probability)
from scripts import m6
from scripts.create_m6_scenarios import build


class FakeEngine:
    def __init__(self, interrupt_at=None):
        self.calls = 0
        self.interrupt_at = interrupt_at

    def ensure_loaded(self):
        pass

    def generate(self, task):
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt()
        return {'answer': '临时QA回答' + str(task['variant']), 'finish_reason': 'eos', 'output_tokens': 3}


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m6.read_json(m6.CONFIG)
        self.cases = build()

    def test_original_plan_has_270_cases_and_balanced_splits(self):
        stats = validate_scenarios(self.cases, self.cfg)
        self.assertEqual(stats['counts'], {'train': 180, 'dev': 30, 'test': 60})
        self.assertEqual(stats['cases'], 270)
        self.assertGreater(stats['contextual'], 0)

    def test_family_leak_and_answer_in_scenario_are_rejected(self):
        cases = deepcopy(self.cases)
        cases[180]['family_id'] = cases[0]['family_id']
        with self.assertRaisesRegex(ValueError, '跨集合'):
            validate_scenarios(cases, self.cfg)
        cases = deepcopy(self.cases)
        cases[0]['chosen'] = '冒充审核答案'
        with self.assertRaisesRegex(ValueError, '冒充'):
            validate_scenarios(cases, self.cfg)

    def test_test_split_cannot_generate_preferences_and_expectation_not_in_input(self):
        with self.assertRaisesRegex(ValueError, '最终测试'):
            build_candidate_tasks(self.cases, self.cfg, '系统', 'test', 1)
        tasks = build_candidate_tasks(self.cases, self.cfg, '系统', 'train', 20)
        self.assertEqual(len(tasks), 40)
        for i in range(0, 40, 2):
            self.assertEqual(tasks[i]['messages'], tasks[i + 1]['messages'])
            self.assertNotEqual(tasks[i]['seed'], tasks[i + 1]['seed'])
            self.assertNotIn('expectation', tasks[i])
        self.assertEqual(tasks, build_candidate_tasks(self.cases, self.cfg, '系统', 'train', 20))

    def test_interrupted_candidate_collection_resumes_without_duplicates(self):
        tasks = build_candidate_tasks(self.cases, self.cfg, '系统', 'train', 3)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'qa.jsonl'
            with self.assertRaises(KeyboardInterrupt):
                m6.generate_candidates(path, tasks, tasks, {}, FakeEngine(interrupt_at=3))
            self.assertEqual(len(m6.read_candidates(path, tasks, {})), 2)
            with self.assertRaisesRegex(ValueError, 'resume'):
                m6.generate_candidates(path, tasks, tasks, {}, FakeEngine())
            engine = FakeEngine()
            rows = m6.generate_candidates(path, tasks, tasks, {}, engine, resume=True)
            self.assertEqual((len(rows), engine.calls), (6, 4))
            cached = FakeEngine()
            m6.generate_candidates(path, tasks, tasks, {}, cached)
            self.assertEqual(cached.calls, 0)

    def test_unknown_duplicate_corrupt_or_changed_source_candidate_rejected(self):
        tasks = build_candidate_tasks(self.cases, self.cfg, '系统', 'train', 1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'qa.jsonl'
            m6.generate_candidates(path, tasks, tasks, {}, FakeEngine())
            original = path.read_text()
            with self.assertRaisesRegex(ValueError, '来源'):
                m6.read_candidates(path, tasks, {'other': 1})
            path.write_text(original + original.splitlines()[0] + '\n')
            with self.assertRaisesRegex(ValueError, '重复'):
                m6.read_candidates(path, tasks, {})
            path.write_text(original + '{')
            with self.assertRaisesRegex(ValueError, '损坏'):
                m6.read_candidates(path, tasks, {})

    def test_review_pack_excludes_test_and_flags_duplicate_or_failed(self):
        tasks = build_candidate_tasks(self.cases, self.cfg, '系统', 'train', 2)
        rows = [{**task, 'answer': '同一回答', 'status': 'ok', 'finish_reason': 'eos'} for task in tasks]
        rows[2].update(status='error', answer='', finish_reason='error')
        result = m6.build_review_items(rows, self.cases, 7)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(not item['reviewable'] for item, key in result))
        self.assertEqual(result, m6.build_review_items(rows, self.cases, 7))

    def valid_review(self):
        item = {'case_id': 'temporary-qa', 'candidates_sha256': 'hash', 'category': 'technical',
                'reviewable': True, 'left_answer': '正确QA', 'right_answer': '错误QA'}
        row = {'case_id': item['case_id'], 'candidates_sha256': 'hash', 'decision': 'accept',
               'preference': 'left', 'reason': '临时QA说明', 'reviewer': 'QA，不是真实评分',
               'confidence': 'high', 'swap_consistent': True, 'technical_verified': True,
               'chosen_voice': 3, 'chosen_persona_context': 4, 'chosen_usefulness': 4}
        return row, item

    def test_no_low_quality_tie_unverified_or_stale_pair_can_be_accepted(self):
        row, item = self.valid_review()
        self.assertEqual(validate_decision(row, item)['chosen'], item['left_answer'])
        for field, value in [('chosen_usefulness', 2), ('confidence', 'low'), ('preference', 'tie'),
                             ('swap_consistent', False), ('technical_verified', False),
                             ('candidates_sha256', 'stale')]:
            bad = {**row, field: value}
            with self.assertRaises(ValueError):
                validate_decision(bad, item)

    def test_revision_keeps_rejected_and_records_source_and_reject_creates_no_pair(self):
        row, item = self.valid_review()
        row.update(decision='revise', revised_chosen='修订的QA答案', revision_source='QA人工式修改')
        pair = validate_decision(row, item)
        self.assertEqual(pair['chosen'], '修订的QA答案')
        self.assertEqual(pair['rejected'], item['right_answer'])
        self.assertTrue(pair['chosen_is_revision'])
        row['decision'] = 'reject'
        self.assertIsNone(validate_decision(row, item))

    def test_export_binds_real_candidate_text_and_rejects_tampered_category(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m6.ROOT/'models/qwen2.5-1.5b-instruct', local_files_only=True)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            result, data, private = root/'results', root/'data', root/'private'
            result.mkdir(); data.mkdir()
            prepared = (self.cfg, self.cases, {}, {}, None, 'QA系统', {'conditions': {}})
            tasks = build_candidate_tasks(self.cases, self.cfg, 'QA系统', 'train', 6)
            known = m6.all_tasks(self.cases, self.cfg, 'QA系统')
            m6.generate_candidates(result/'candidates.jsonl', tasks, known, {}, FakeEngine())
            with patch.object(m6, 'ROOT', root), patch.object(m6, 'RESULTS', result), \
                 patch.object(m6, 'DATA', data), patch.object(m6, 'PRIVATE', private), \
                 patch.object(m6, 'prepare', return_value=prepared), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=tok):
                m6.review_pack('train')
                package_path = next((result/'review').glob('*.json'))
                package = m6.read_json(package_path)
                item = next(i for i in package['items'] if i['category']=='technical')
                row = {'case_id':item['case_id'], 'candidates_sha256':item['candidates_sha256'],
                       'pack_id':package['pack_id'], 'decision':'accept', 'preference':'left',
                       'reviewer':'临时QA评分者', 'reason':'临时QA例子，不是正式偏好', 'confidence':'high',
                       'swap_consistent':True, 'technical_verified':True, 'chosen_voice':3,
                       'chosen_persona_context':4, 'chosen_usefulness':4}
                (data/'review_decisions.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
                m6.export_pairs()
                exported=m6.read_jsonl(next((data/'pairs').glob('*/train.jsonl')))
                self.assertEqual(len(exported),1)
                self.assertEqual(exported[0]['chosen'],item['left_answer'])
                item['category']='daily'
                m6.dump_json(package_path,package)
                with self.assertRaisesRegex(ValueError,'划分'):
                    m6.export_pairs()

    def test_real_tokenizer_keeps_one_eos_and_rejects_truncation(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m6.ROOT/'models/qwen2.5-1.5b-instruct', local_files_only=True)
        messages = [{'role': 'system', 'content': '临时QA系统'}, {'role': 'user', 'content': '临时QA问题'}]
        encoded = encode_preference(tok, messages, '临时QA较好回复。', '临时QA较差回复。', 320)
        for side in ('chosen', 'rejected'):
            self.assertEqual(encoded[side+'_input_ids'].count(tok.eos_token_id), 1)
            self.assertEqual(encoded[side+'_input_ids'][-1], tok.eos_token_id)
        with self.assertRaisesRegex(ValueError, '截断'):
            encode_preference(tok, messages, '临时QA較好回复。', '临时QA较差回复。', 3)

    def test_causal_probability_ignores_prompt_and_padding_but_supervises_answer(self):
        import torch
        torch.manual_seed(9)
        logits = torch.randn(1, 5, 10, requires_grad=True)
        ids = torch.tensor([[1, 2, 3, 4, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 0]])
        score = completion_log_probability(logits, ids, mask)
        manual = logits[0, 1].log_softmax(0)[3] + logits[0, 2].log_softmax(0)[4]
        self.assertTrue(torch.allclose(score[0], manual))
        score.backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 3:].abs().sum()), 0)

    def test_dpo_initial_log_two_and_gradient_direction(self):
        import torch
        p = torch.tensor([-5.], requires_grad=True)
        n = torch.tensor([-6.], requires_grad=True)
        loss, margin = dpo_loss(p, n, p.detach(), n.detach())
        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)
        self.assertEqual(float(margin.detach()), 0)
        loss.mean().backward()
        self.assertLess(float(p.grad), 0)
        self.assertGreater(float(n.grad), 0)

    @unittest.skipUnless(importlib.util.find_spec('trl'), '实际TRL测试需.venv-m6')
    def test_real_trl_toy_dpo_updates_only_policy_and_keeps_sft_reference(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, PeftModel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from trl import DPOConfig, DPOTrainer
        torch.manual_seed(11)
        vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'prompt': 3, 'good': 4, 'bad': 5}
        tok = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        tok.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
        base = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_layer=1, n_head=1, n_embd=8, n_positions=32,
            bos_token_id=1, eos_token_id=1, pad_token_id=0))
        sft = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'], task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, p in sft.named_parameters():
                if 'lora_B' in name:
                    p.fill_(0.05)  # 模拟已有非零SFT，不能关闭adapter拿裸基模作reference。
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)
            sft.save_pretrained(path/'sft', save_embedding_layers=False)
            plain = sft.unload()
            model = PeftModel.from_pretrained(plain, path/'sft', adapter_name='policy', is_trainable=True)
            model.load_adapter(path/'sft', adapter_name='reference', is_trainable=False)
            model.set_adapter('policy')
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            data = Dataset.from_dict({'prompt': ['prompt ']*2, 'chosen': ['good']*2, 'rejected': ['bad']*2})
            args = DPOConfig(output_dir=str(path/'run'), use_cpu=True, report_to='none', max_steps=1,
                learning_rate=1e-3, per_device_train_batch_size=1, gradient_checkpointing=False,
                save_strategy='no', disable_tqdm=True, dataloader_pin_memory=False,
                model_adapter_name='policy', ref_adapter_name='reference', precompute_ref_log_probs=True,
                max_prompt_length=None, max_completion_length=None, max_length=32,
                optim='adamw_torch', bf16=False, fp16=False)
            trainer = DPOTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=data)
            batch = next(iter(trainer.get_train_dataloader()))
            pc, pr = trainer.concatenated_forward(model, batch)['chosen_logps'], trainer.concatenated_forward(model, batch)['rejected_logps']
            rc, rr = trainer.compute_ref_log_probs(batch)
            self.assertTrue(torch.allclose(pc, rc))
            self.assertTrue(torch.allclose(pr, rr))
            trainer.train()
            changed = []
            for name, p in model.named_parameters():
                if not torch.equal(before[name], p):
                    changed.append(name)
                    self.assertIn('lora_', name)
                    self.assertIn('.policy.', name)
                if '.reference.' in name:
                    self.assertFalse(p.requires_grad)
            self.assertTrue(changed)


if __name__ == '__main__':
    unittest.main()
