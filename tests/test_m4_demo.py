"""M4：完整轮次裁剪、真实Adapter隔离、会话状态与失败恢复。"""
from contextlib import contextmanager
from copy import deepcopy
from contextlib import redirect_stdout
import asyncio
import io
import unittest
from unittest.mock import patch

import torch
import demo


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(demo.ROOT / 'models/qwen2.5-1.5b-instruct', local_files_only=True)

    def test_drops_only_old_complete_turns_and_preserves_system_and_current_question(self):
        history = [{'role': 'user', 'content': '旧问题' * 50}, {'role': 'assistant', 'content': '旧回复' * 50},
                   {'role': 'user', 'content': '小猫叫薄荷。'}, {'role': 'assistant', 'content': '我听见了。'}]
        original = deepcopy(history)
        _, _, _, budget = demo.prepare_messages(self.tokenizer, '固定角色', history[2:], '小猫叫什么？', 768)
        messages, kept, dropped, count = demo.prepare_messages(
            self.tokenizer, '固定角色', history, '小猫叫什么？', budget)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, history[2:])
        self.assertEqual(messages[0], {'role': 'system', 'content': '固定角色'})
        self.assertEqual(messages[-1]['content'], '小猫叫什么？')
        self.assertLessEqual(count, budget)
        self.assertEqual(history, original)

    def test_rejects_overlong_current_question_and_incomplete_history(self):
        with self.assertRaisesRegex(ValueError, '没有被截断'):
            demo.prepare_messages(self.tokenizer, '固定角色', [], '很长' * 800, 768)
        with self.assertRaisesRegex(ValueError, '完整问答'):
            demo.prepare_messages(self.tokenizer, '固定角色', [{'role': 'user', 'content': '未完成'}], '你好', 768)
        with self.assertRaisesRegex(ValueError, '输入一句话'):
            demo.prepare_messages(self.tokenizer, '固定角色', [], '  ', 768)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, return_tensors=None, **kwargs):
        return torch.tensor([[1, 3]]) if return_tensors else [1, 3]

    def decode(self, tokens, **kwargs):
        return '原模型回复' if int(tokens[0]) == 4 else '微调模型回复'


class FakeModel:
    def __init__(self):
        self.disabled = False
        self.calls = []
        self.fail = False

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def generate(self, **kwargs):
        self.calls.append({'disabled': self.disabled, 'grad_enabled': torch.is_grad_enabled(), **kwargs})
        if self.fail:
            raise RuntimeError('模拟推理失败')
        new = torch.tensor([[4 if self.disabled else 5, 2]])
        return torch.cat([kwargs['input_ids'], new], dim=1)

    def requires_grad_(self, enabled):
        self.grad_enabled = enabled
        return self


class EngineTests(unittest.TestCase):
    def engine(self):
        cfg = demo.read_json(demo.DEFAULT_CONFIG)
        prompts = {'short': '短角色', 'strong_examples': '详细角色加示例'}
        with patch.object(demo, 'validate_assets', return_value=(cfg, {}, None, prompts)):
            engine = demo.ChatEngine()
        engine.model, engine.tokenizer, engine.device = FakeModel(), FakeTokenizer(), 'cpu'
        return engine

    def test_comparison_shares_inputs_and_decoding_and_disables_only_base_adapter(self):
        engine = self.engine()
        result = engine.compare('你好')
        base, lora = engine.model.calls
        self.assertTrue(base['disabled'])
        self.assertFalse(lora['disabled'])
        self.assertFalse(base['grad_enabled'])
        self.assertFalse(lora['grad_enabled'])
        for key in ('input_ids', 'attention_mask'):
            self.assertTrue(torch.equal(base[key], lora[key]))
        for key in ('max_new_tokens', 'do_sample', 'pad_token_id', 'eos_token_id', 'use_cache'):
            self.assertEqual(base[key], lora[key])
        self.assertEqual(result['messages'][0]['content'], '详细角色加示例')
        self.assertEqual(result['base']['answer'], '原模型回复')
        self.assertEqual(result['lora']['answer'], '微调模型回复')
        self.assertEqual(result['base']['output_tokens'], 2)
        self.assertFalse(engine.model.disabled)

    def test_failed_base_inference_restores_adapter_and_preserves_history(self):
        engine = self.engine()
        history = [{'role': 'user', 'content': '前文'}, {'role': 'assistant', 'content': '回复'}]
        original = deepcopy(history)
        engine.model.fail = True
        with self.assertRaisesRegex(RuntimeError, '模拟推理失败'):
            engine.reply('继续', history, mode='base')
        self.assertFalse(engine.model.disabled)
        self.assertEqual(history, original)
        engine.model.fail = False
        result = engine.reply('继续', history, mode='lora')
        self.assertEqual(result['history'][-1]['content'], '微调模型回复')
        self.assertEqual(history, original)

    def test_chat_reuses_recent_history_but_comparison_is_independent(self):
        engine = self.engine()
        first = engine.reply('小猫叫薄荷。')
        second = engine.reply('它叫什么？', first['history'])
        self.assertEqual(second['messages'][1:-1], first['history'])
        comparison = engine.compare('独立问题')
        self.assertEqual([m['role'] for m in comparison['messages']], ['system', 'user'])

    def test_actual_saved_m3_assets_pass_validation_without_loading_model(self):
        cfg, train_cfg, adapter, prompts = demo.validate_assets()
        self.assertEqual(train_cfg['run_id'], 'm3_learning_01')
        self.assertTrue((adapter / 'adapter_model.safetensors').exists())

    def test_real_peft_disable_adapter_exit_cannot_leave_trainable_parameters(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, get_peft_model
        engine = self.engine()
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        ))
        engine.model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        engine.model.eval().requires_grad_(False)
        engine.cfg['max_new_tokens'] = 2
        for mode in ['base', 'lora']:
            engine.reply('测试输入', mode=mode)
            self.assertFalse(any(p.requires_grad for p in engine.model.parameters()))


class UIEventTests(unittest.TestCase):
    def setUp(self):
        self.engine = EngineTests().engine()
        async def construct():
            return demo.build_app(self.engine)
        self.app = asyncio.run(construct())
        self.callbacks = {fn.fn.__name__: fn for fn in self.app.fns.values()}
        self.history = next(c for c in self.app.blocks.values() if c.__class__.__name__ == 'State')
        self.chatbot = next(c for c in self.app.blocks.values() if c.__class__.__name__ == 'Chatbot')
        self.message = next(c for c in self.app.blocks.values() if getattr(c, 'label', None) == '你的消息')
        self.mode = next(c for c in self.app.blocks.values() if getattr(c, 'label', None) == '当前模型')

    def test_chat_event_disables_controls_then_commits_reply_and_restores_controls(self):
        event = self.callbacks['send_chat']
        generator = event.fn('你好', [], 'lora')
        initial, completed = next(generator), next(generator)
        self.assertFalse(initial[self.message]['interactive'])
        self.assertFalse(initial[self.mode]['interactive'])
        self.assertEqual(completed[self.history][-1]['content'], '微调模型回复')
        self.assertEqual(completed[self.chatbot], completed[self.history])
        self.assertTrue(completed[self.message]['interactive'])
        self.assertTrue(completed[self.mode]['interactive'])
        self.assertEqual(completed[self.message]['value'], '')
        self.assertEqual(event.concurrency_id, 'model')
        self.assertEqual(event.concurrency_limit, 1)
        self.assertEqual(self.app.config['dependencies'][0]['trigger_mode'], 'once')

    def test_new_session_and_mode_change_clear_history(self):
        fn = self.callbacks['reset_chat'].fn
        for selected in ['base', 'lora']:
            result = fn(selected)
            self.assertEqual(result[self.history], [])
            self.assertEqual(result[self.chatbot], [])
            self.assertEqual(result[self.message], '')
        clear_events = [event for event in self.app.config['dependencies']
                        if [self.chatbot._id, 'clear'] in [list(target) for target in event['targets']]]
        self.assertEqual(len(clear_events), 1)

    def test_failed_generation_keeps_history_and_reenables_controls(self):
        self.engine.model.fail = True
        previous = [{'role': 'user', 'content': '前文'}, {'role': 'assistant', 'content': '回复'}]
        with redirect_stdout(io.StringIO()):
            stages = list(self.callbacks['send_chat'].fn('继续', previous, 'base'))
        final = stages[-1]
        self.assertEqual(final[self.chatbot], previous)
        self.assertNotIn(self.history, final)
        self.assertTrue(final[self.message]['interactive'])
        self.assertTrue(final[self.mode]['interactive'])
        self.assertFalse(self.engine.model.disabled)

    def test_comparison_has_same_queue_and_does_not_return_chat_history(self):
        event = self.callbacks['compare_models']
        results = list(event.fn('比较问题'))
        self.assertNotIn(self.history, results[-1])
        self.assertEqual(event.concurrency_id, 'model')
        self.assertTrue(results[-1][self.message]['interactive'])
        self.assertEqual(len(self.engine.model.calls), 2)


if __name__ == '__main__':
    unittest.main()
