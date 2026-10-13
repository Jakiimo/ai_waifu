"""M2 数据整理：场景展开、去重、隔离检查、真实 Tokenizer 标签验证。

运行：.venv/bin/python prepare_data.py
只准备数据，不更新模型参数。已冻结时默认仅核对文件，不覆盖。
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata
from training_data import encode_conversation

ROOT = Path(__file__).resolve().parent
M2 = ROOT / "data" / "m2"
RESULTS = ROOT / "results" / "m2"
CONFIG = ROOT / "configs" / "m2.json"
PROMPTS = ROOT / "configs" / "m2_prompts.json"
RAW = M2 / "raw_cards.jsonl"
DEV = M2 / "dev_scenarios.jsonl"
TEST = ROOT / "data" / "test_scenarios.jsonl"
TRAIN = M2 / "train.jsonl"
FROZEN = M2 / "frozen_manifest.json"
MODEL = ROOT / "models" / "qwen2.5-1.5b-instruct"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


def grams(text):
    text = normalize(text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def similarity(left, right):
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def check_cases(cases, name):
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError(f"{name} 中出现重复 id。")
    for case in cases:
        messages = case.get("messages", [])
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"{case['id']} 必须以 user 消息结尾。")
        for i, message in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            if message.get("role") != expected or not message.get("content", "").strip():
                raise ValueError(f"{case['id']} 的消息顺序或内容错误。")


def expand_cards(cards):
    """同族问法始终留在训练区；重复输入保留一条，冲突回答则失败。"""
    if len({card["family_id"] for card in cards}) != len(cards):
        raise ValueError("原始场景族 id 重复。")
    rows, seen, removed = [], {}, []
    for card in cards:
        if not card["assistant"].strip() or not card["user_variants"]:
            raise ValueError(f"{card['family_id']} 目标或问法为空。")
        for index, user in enumerate(card["user_variants"], 1):
            if not isinstance(user, str) or not normalize(user):
                raise ValueError(f"{card['family_id']} 输入为空。")
            history = card.get("history", [])
            check_cases([{"id": card["family_id"], "messages": history + [{"role": "user", "content": user}]}], "训练前文")
            key = context_key(history + [{"role": "user", "content": user}])
            if key in seen:
                if seen[key]["messages"][-1]["content"] != card["assistant"]:
                    raise ValueError("相同输入出现不同目标回答，请人工决定保留哪一条。")
                removed.append({"family_id": card["family_id"], "reason": "duplicate_input"})
                continue
            row = {
                "id": f"{card['family_id']}-v{index}", "family_id": card["family_id"],
                "category": card["category"], "source": card["source"], "review": card["review"],
                "messages": history + [{"role": "user", "content": user.strip()},
                             {"role": "assistant", "content": card["assistant"].strip()}],
            }
            seen[key] = row
            rows.append(row)
    return rows, removed


def context_messages(row):
    messages = row["messages"]
    return messages[:-1] if messages[-1]["role"] == "assistant" else messages


def context_key(messages):
    return "\n".join(m["role"] + ":" + normalize(m["content"]) for m in messages)


def context_text(row):
    return "\n".join(m["role"] + ":" + m["content"] for m in context_messages(row))


def check_isolation(train, dev, tests, threshold):
    groups = {"train": train, "dev": dev, "test": tests}
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = {r["family_id"] for r in groups[left]} & {r["family_id"] for r in groups[right]}
        if overlap:
            raise ValueError(f"{left}/{right} 场景族交叉：{sorted(overlap)}")
    near, closest = [], []
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        for a in groups[left]:
            text_a = context_text(a)
            for b in groups[right]:
                text_b = context_text(b)
                if context_key(context_messages(a)) == context_key(context_messages(b)):
                    raise ValueError(f"{a['id']}/{b['id']} 跨集合输入重复。")
                score = similarity(text_a, text_b)
                item = {"left": a["id"], "right": b["id"], "similarity": round(score, 4)}
                closest.append(item)
                if score >= threshold:
                    near.append(item)
    # 同一训练族的改写是有意增强；不同训练族仍检查近重复。
    representatives = {}
    for row in train:
        representatives.setdefault(row["family_id"], []).append(row)
    families = list(representatives)
    for i, left in enumerate(families):
        for right in families[i + 1:]:
            for a in representatives[left]:
                for b in representatives[right]:
                    score = similarity(context_text(a), context_text(b))
                    if score >= threshold:
                        near.append({"left": a["id"], "right": b["id"], "similarity": round(score, 4)})
    if near:
        raise ValueError("发现需审核的近重复输入：" + json.dumps(near[:10], ensure_ascii=False))
    return {"threshold": threshold, "near_pairs_at_threshold": len(near),
            "closest_cross_split_pairs": sorted(closest, key=lambda x: x["similarity"], reverse=True)[:10],
            "limitation": "字符三元组检查，不是语义泄漏证明；同族改写有意保留，需人工抽样。"}


def frozen_paths():
    return [CONFIG, PROMPTS, RAW, TRAIN, DEV, TEST, ROOT / "configs" / "m2_rubric.json", ROOT / "docs" / "凛音_角色卡.md",
            ROOT / "prepare_data.py", ROOT / "training_data.py", ROOT / "scripts" / "m2.py"]


def verify_frozen():
    if not FROZEN.exists():
        raise ValueError("尚未冻结。先运行开发集实验，再执行 freeze。")
    manifest = read_json(FROZEN)
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"冻结文件已变化：{relative}。需要创建新版本，不能冒用旧实验。")
    if "model_revision" in manifest:
        model = read_json(ROOT / "models" / "model_manifest.json")
        if model["revision"] != manifest["model_revision"] or model["model_id"] != manifest["model_id"]:
            raise ValueError("本地模型版本与冻结实验不同。")
    return manifest


def prepare():
    if FROZEN.exists():
        verify_frozen()
        print("M2 已冻结，文件校验通过；未覆盖数据。")
        return read_json(RESULTS / "data_report.json")
    cfg, prompts = read_json(CONFIG), read_json(PROMPTS)
    cards, dev, tests = read_jsonl(RAW), read_jsonl(DEV), read_jsonl(TEST)
    rubric = read_json(ROOT / "configs" / "m2_rubric.json")
    if rubric["version"] != cfg["rubric_version"]:
        raise ValueError("评分规则与当前配置不一致。")
    if cfg["selected_strong_prompt"] not in prompts:
        raise ValueError("选定 Prompt 不存在。")
    for case in dev + tests:
        if case.get("dataset_version") != cfg["dataset_version"] or case.get("rubric_version") != cfg["rubric_version"]:
            raise ValueError("开发/测试样本版本过期：" + case["id"])

    check_cases(dev, "开发集")
    check_cases(tests, "测试集")
    if not 40 <= len(tests) <= 60:
        raise ValueError("测试集数量需在 40–60 个独立场景内。")
    train, removed = expand_cards(cards)
    isolation = check_isolation(train, dev, tests, cfg["near_duplicate_threshold"])

    # 使用 M1 的真实 Tokenizer 和标签检查函数，不加载 GPU 模型。
    sys.path.insert(0, str(ROOT / "scripts"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    token_checks = []
    for row in train:
        encoded, counts = encode_conversation(tokenizer, prompts["short"], row["messages"], cfg["max_length"])
        assert len(encoded["labels"]) == cfg["max_length"]
        token_checks.append({"id": row["id"], **counts})
    for case in dev:
        encode_conversation(tokenizer, prompts["short"], case["messages"] + [
            {"role": "assistant", "content": case["reference_answer"]}], cfg["max_length"])
    inference_lengths = []
    for name, prompt in prompts.items():
        for case in dev + tests:
            messages = [{"role": "system", "content": prompt}] + case["messages"]
            length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            if length > cfg["max_input_tokens"]:
                raise ValueError(f"{case['id']}/{name} 输入 {length} token，超过推理上限。")
            inference_lengths.append(length)
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train), encoding="utf-8")
    report = {
        "dataset_version": cfg["dataset_version"], "persona": cfg["persona"],
        "raw_families": len(cards), "raw_variants": sum(len(c["user_variants"]) for c in cards),
        "contextual_examples": sum(len(r["messages"]) > 2 for r in train),
        "single_turn_examples": sum(len(r["messages"]) == 2 for r in train),
        "rubric_version": cfg["rubric_version"],
        "training_examples": len(train), "training_families": len({r["family_id"] for r in train}),
        "unique_target_answers": len({r["messages"][-1]["content"] for r in train}),
        "dev_scenarios": len(dev), "test_scenarios": len(tests), "removed": removed,
        "train_categories": dict(Counter(r["category"] for r in train)),
        "dev_categories": dict(Counter(r["category"] for r in dev)),
        "test_categories": dict(Counter(r["category"] for r in tests)),
        "max_training_tokens": max(c["prompt_tokens"] + c["answer_tokens"] for c in token_checks),
        "max_inference_input_tokens": max(inference_lengths),
        "isolation": isolation, "token_checks": token_checks,
        "source": "project_authored_ai_assisted", "human_review": "pending",
        "files": {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths()},
    }
    dump_json(RESULTS / "data_report.json", report)
    # 每类六个不同场景族，保留人工已填写结果，不随机覆盖抽查表。
    rng = random.Random(cfg["seed"])
    sample = []
    for category in sorted({r["category"] for r in train}):
        representatives = {}
        for row in train:
            if row["category"] == category:
                representatives.setdefault(row["family_id"], row)
        sample.extend(rng.sample(list(representatives.values()), 6))
    sample.sort(key=lambda r: (r["category"], r["id"]))
    review_path = RESULTS / "review_samples.csv"
    if not review_path.exists():
        with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "category", "history", "user", "assistant", "human_verdict", "notes"])
            writer.writeheader()
            for row in sample:
                writer.writerow({"id": row["id"], "category": row["category"],
                    "history": " / ".join(m["role"] + ": " + m["content"] for m in row["messages"][:-2]),
                    "user": row["messages"][-2]["content"], "assistant": row["messages"][-1]["content"],
                    "human_verdict": "pending", "notes": ""})
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in train}
    lines = ["# 凛音：傲娇 v2 台词抽查", "", "30 个不同训练场景族；是原创训练目标，不是模型效果。CSV 可填写 pass/revise/reject。", ""]
    for item in review:
        row = by_id[item["id"]]
        lines.extend([f"## {row['id']} ({row['category']})", ""])
        for message in row["messages"]:
            lines.extend([("凛音" if message["role"] == "assistant" else "用户") + "：" + message["content"], ""])
        lines.extend(["人工意见：" + item["human_verdict"] + "；" + item["notes"], ""])
    (RESULTS / "review_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["training_examples", "training_families", "unique_target_answers",
          "dev_scenarios", "test_scenarios", "max_training_tokens", "max_inference_input_tokens", "human_review"]},
          ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    try:
        prepare()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"数据准备未通过：{error}")
