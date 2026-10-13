"""新多轮候选数据的监督边界；验证历史不会被误当成本次目标。"""
import importlib.util
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tsundere", ROOT / "scripts/create_tsundere_v2.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class ConversationLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen2.5-1.5b-instruct", local_files_only=True)
        cls.system = (ROOT / "data/m2_tsundere_v2/short_prompt.txt").read_text().strip()
        cls.rows = [json.loads(line) for line in (ROOT / "data/m2_tsundere_v2/train_candidate.jsonl").read_text().splitlines()]

    def test_all_candidates_supervise_only_final_reply(self):
        self.assertEqual(len(self.rows), 360)
        self.assertEqual(sum(len(row["messages"]) > 2 for row in self.rows), 60)
        for row in self.rows:
            data, counts = t.encode_conversation(self.tokenizer, self.system, row["messages"], 256)
            active = [token for token in data["labels"] if token != -100]
            text = self.tokenizer.decode(active)
            self.assertEqual(text.split(self.tokenizer.eos_token)[0], row["messages"][-1]["content"])
            self.assertTrue(all(label == -100 for label in data["labels"][:counts["prompt_tokens"]]))
            self.assertIn(self.tokenizer.eos_token_id, active)
            self.assertEqual(len(data["labels"]), 256)

    def test_empty_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "内容为空"):
            t.encode_conversation(self.tokenizer, self.system,
                                  [{"role": "user", "content": "你好"}, {"role": "assistant", "content": " "}], 256)

    def test_long_history_does_not_silently_remove_target(self):
        messages = [{"role": "user", "content": "很长的历史消息" * 300},
                    {"role": "assistant", "content": "收到"},
                    {"role": "user", "content": "继续说"},
                    {"role": "assistant", "content": "当前目标回答"}]
        with self.assertRaisesRegex(ValueError, "拒绝截断"):
            t.encode_conversation(self.tokenizer, self.system, messages, 256)


if __name__ == "__main__":
    unittest.main()
