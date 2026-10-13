"""用真实 Qwen Tokenizer 检查训练监督范围，不加载模型或运行训练。"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m1", ROOT / "scripts" / "m1.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)


class LabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(m1.MODEL_PATH, local_files_only=True)
        cls.cfg = m1.read_json(m1.CONFIG_PATH)

    def test_real_samples_only_supervise_answer_and_end_marker(self):
        dataset, counts = m1.encoded_data(self.tokenizer, self.cfg)
        self.assertEqual(len(dataset), 12)
        for row, count in zip(dataset, counts):
            prompt_end = count["prompt_tokens"]
            target_end = prompt_end + count["answer_tokens"]
            self.assertTrue(all(label == -100 for label in row["labels"][:prompt_end]))
            self.assertEqual(row["labels"][prompt_end:target_end], row["input_ids"][prompt_end:target_end])
            self.assertTrue(all(label == -100 for label in row["labels"][target_end:]))
            self.assertTrue(all(mask == 0 for mask in row["attention_mask"][target_end:]))
            self.assertIn(self.tokenizer.eos_token_id, row["labels"][prompt_end:target_end])
            self.assertEqual(len(row["input_ids"]), self.cfg["max_length"])

    def test_overlong_example_is_rejected_instead_of_losing_answer(self):
        example = {"id": "long", "user": "很长的问题" * 200, "assistant": "一个回答。"}
        with self.assertRaisesRegex(ValueError, "超过 max_length"):
            m1.encode_example(self.tokenizer, self.cfg, example)

    def test_empty_answer_is_not_an_eos_only_training_example(self):
        example = {"id": "empty", "user": "你好", "assistant": "  "}
        with self.assertRaisesRegex(ValueError, "assistant"):
            m1.encode_example(self.tokenizer, self.cfg, example)

    def test_template_mismatch_is_rejected(self):
        class BrokenTemplate:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2] if kwargs["add_generation_prompt"] else [3, 4, 5]
        with self.assertRaisesRegex(ValueError, "回答起点不一致"):
            m1.encode_example(BrokenTemplate(), self.cfg,
                              {"id": "bad", "user": "你好", "assistant": "你好。"})


if __name__ == "__main__":
    unittest.main()
