"""M2 的泄漏、去重和冻结保护测试；使用临时文件，不改正式测试集。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "prepare_data.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def row(id, family, text):
    return {"id": id, "family_id": family, "messages": [{"role": "user", "content": text}]}


class DataIntegrityTests(unittest.TestCase):
    def test_family_leak_is_rejected_even_with_different_wording(self):
        with self.assertRaisesRegex(ValueError, "场景族交叉"):
            p.check_isolation([row("a", "same", "今天下雨了")],
                              [row("b", "same", "电脑如何关机")], [], 0.65)

    def test_near_duplicate_across_splits_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "近重复"):
            p.check_isolation([row("a", "a", "我今天学习了一下午但感觉没有什么进展")],
                              [], [row("b", "b", "我今天学习了一下午但感觉没有什么进步")], 0.65)

    def test_same_input_with_conflicting_answers_is_rejected(self):
        cards = [{"family_id": "a", "category": "daily", "user_variants": ["你好", "你好！"],
                  "assistant": "好呀", "source": "test", "review": "test"},
                 {"family_id": "b", "category": "daily", "user_variants": ["你好"],
                  "assistant": "再见", "source": "test", "review": "test"}]
        with self.assertRaisesRegex(ValueError, "不同目标回答"):
            p.expand_cards(cards)

    def test_duplicate_input_is_removed_without_changing_group(self):
        cards = [{"family_id": "a", "category": "daily", "user_variants": ["你好", "你好！"],
                  "assistant": "好呀", "source": "test", "review": "test"}]
        rows, removed = p.expand_cards(cards)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["family_id"], "a")
        self.assertEqual(len(removed), 1)

    def test_frozen_file_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target = root / "test.jsonl"
            target.write_text("original")
            manifest = root / "frozen.json"
            p.dump_json(manifest, {"files": {"test.jsonl": p.digest(target)}})
            with patch.object(p, "ROOT", root), patch.object(p, "FROZEN", manifest):
                p.verify_frozen()
                target.write_text("changed")
                with self.assertRaisesRegex(ValueError, "冻结文件已变化"):
                    p.verify_frozen()

    def test_test_cases_cannot_end_with_reference_answer(self):
        cases = [{"id": "a", "messages": [{"role": "user", "content": "问题"},
                                           {"role": "assistant", "content": "答案"}]}]
        with self.assertRaisesRegex(ValueError, "user 消息结尾"):
            p.check_cases(cases, "test")


if __name__ == "__main__":
    unittest.main()
