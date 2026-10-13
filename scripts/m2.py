"""M2 开发集推理、Prompt 选择与实验文件冻结；不会训练 LoRA。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_data import (
    CONFIG, DEV, FROZEN, PROMPTS, RESULTS, digest, dump_json, frozen_paths,
    prepare, read_json, read_jsonl, verify_frozen, review_state,
)


def measure_style(answer):
    """仅自动线索，不冒充人工评分；中文标点近似切句，不把换行当句号。"""
    sentences = [part for part in re.split(r"[。！？!?]+", answer) if part.strip()]
    return {
        "approx_sentences": len(sentences),
        "undesired_address": any(word in answer for word in ("主人", "亲爱的", "宝贝")),
        "repeated_punctuation": bool(re.search(r"[!！?？]{3,}", answer)),
        "warning": "口癖、否认句、游戏词的多少不代表傲娇质量；线索不代替人工评分。",
    }


def validate_probe(report):
    """版本名与全部输入证据都匹配，才能复用或冻结模型输出。"""
    cfg = read_json(CONFIG)
    if report.get("dataset_version") != cfg["dataset_version"] or report.get("rubric_version") != cfg["rubric_version"]:
        raise ValueError("开发记录的角色/评分版本过期，不能用于当前 M2。")
    for relative, expected in report.get("input_files", {}).items():
        if digest(ROOT / relative) != expected:
            raise ValueError("开发记录的输入已变化：" + relative)
    expected_paths = {str(p.relative_to(ROOT)) for p in frozen_paths()}
    if set(report.get("input_files", {})) != expected_paths:
        raise ValueError("开发记录缺少完整输入证据。")
    if report.get("model") != read_json(ROOT / "models" / "model_manifest.json"):
        raise ValueError("开发记录的基础模型不一致。")
    records = report.get("records", [])
    pairs = [(r["prompt_name"], r["case_id"]) for r in records]
    expected_pairs = {(name, cid) for name in read_json(PROMPTS) for cid in cfg["probe_ids"]}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("开发输出缺少候选 Prompt/场景，或存在重复记录。")
    return report


def dev_probe():
    if FROZEN.exists():
        manifest = verify_frozen()
        for relative, expected in manifest['evidence_files'].items():
            if digest(ROOT / relative) != expected:
                raise ValueError('冻结实验记录发生变化：' + relative)
        report = validate_probe(read_json(RESULTS / 'prompt_probe.json'))
        print('M2已冻结；开发输出与证据校验通过，复用已有回答，未重新生成。')
        return report
    prepare()
    cfg, prompts = read_json(CONFIG), read_json(PROMPTS)
    if (RESULTS / "prompt_probe.json").exists():
        old = read_json(RESULTS / "prompt_probe.json")
        validate_probe(old)
        print("已有同版本开发集推理记录，未重复生成。")
        return old
    import torch
    from m1 import load_base
    cases = {case["id"]: case for case in read_jsonl(DEV)}
    model, tokenizer, device = load_base()
    if read_json(ROOT / "models" / "model_manifest.json")["model_id"] != cfg["model_id"]:
        raise ValueError("M2 选型与本地基础模型不同。")
    model.eval()
    records = []
    for prompt_name, prompt in prompts.items():
        for case_id in cfg["probe_ids"]:
            case = cases[case_id]
            messages = [{"role": "system", "content": prompt}] + case["messages"]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_tensors="pt").to(device)
            if ids.shape[1] > cfg["max_input_tokens"]:
                raise ValueError("开发输入超过已定义长度。")
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("开发推理发生数值异常。")
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True)
            generated = out[0, ids.shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not answer or not any(char.isalnum() for char in answer):
                raise RuntimeError("开发推理没有有效文字。")
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            eos_ids = set(eos_ids + [tokenizer.eos_token_id])
            finished = int(generated[-1]) in eos_ids
            record = {"prompt_name": prompt_name, "case_id": case_id, "category": case["category"],
                      "messages": messages, "answer": answer, "input_tokens": ids.shape[1],
                      "output_tokens": generated.numel(), "finish_reason": "eos" if finished else "length",
                      "seconds": round(time.perf_counter() - started, 3), "automatic_clues": measure_style(answer)}
            records.append(record)
            print(f"{prompt_name}/{case_id}：{generated.numel()} token，{record['finish_reason']}", flush=True)
    report = {
        "purpose": "development-only prompt inspection, no quality claim",
        "dataset_version": cfg["dataset_version"], "rubric_version": cfg["rubric_version"],
        "persona": cfg["persona"],
        "input_files": {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths()},
        "scope": f"共 {len(cfg['probe_ids'])} 个开发场景、{len(prompts)} 个 Prompt，共 {len(records)} 个回答；未运行测试集。",
        "config_sha256": digest(CONFIG), "prompts_sha256": digest(PROMPTS), "dev_sha256": digest(DEV),
        "device": device, "weight_dtype": str(next(model.parameters()).dtype),
        "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "model": read_json(ROOT / "models" / "model_manifest.json"),
        "decoding": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"]}, "records": records,
    }
    dump_json(RESULTS / "prompt_probe.json", report)
    lines = ["# 凛音：俏皮傲娇 v3 开发集 Prompt 对照", "", "版本：" + report["dataset_version"] + "；评分规则：" + report["rubric_version"], "", report["scope"], "", "这些输出用于选择 Prompt，不是微调结果或最终盲评。", ""]
    for case_id in cfg["probe_ids"]:
        case = cases[case_id]
        lines.extend([f"## {case_id} {case['category']}", ""])
        for message in case["messages"]:
            lines.extend([("凛音（预置前文）" if message["role"] == "assistant" else "用户") + "：" + message["content"], ""])
        for record in records:
            if record["case_id"] == case_id:
                lines.extend([f"### {record['prompt_name']}", "", record["answer"], "",
                              f"输入 {record['input_tokens']} token；输出 {record['output_tokens']} token；结束原因 {record['finish_reason']}。", ""])
    (RESULTS / "prompt_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def freeze():
    if FROZEN.exists():
        verify_frozen()
        print("已有冻结版本且文件一致；未覆盖。")
        return
    prepare()
    cfg = read_json(CONFIG)
    report = validate_probe(read_json(RESULTS / "prompt_probe.json"))
    decision = read_json(RESULTS / "prompt_decision.json")
    if decision.get("ready_to_freeze") is not True:
        raise ValueError("Prompt 选择仍是暂定，当前准备包不能冻结；先处理开发问题。")
    if decision.get("dataset_version") != cfg["dataset_version"]:
        raise ValueError("Prompt 选择说明属于其他数据版本。")
    if decision["selected"] != cfg["selected_strong_prompt"] or decision["probe_sha256"] != digest(RESULTS / "prompt_probe.json"):
        raise ValueError("需要与本次开发输出对应的 Prompt 选择说明。")
    if not any(row["prompt_name"] == decision["selected"] for row in report["records"]):
        raise ValueError("选定 Prompt 没有开发记录。")
    manifest = {
        "dataset_version": cfg["dataset_version"], "persona": cfg["persona"],
        "selected_strong_prompt": decision["selected"], "rubric_version": cfg["rubric_version"],
        "files": {str(path.relative_to(ROOT)): digest(path) for path in frozen_paths()},
        "evidence_files": {str(path.relative_to(ROOT)): digest(path) for path in
                           [RESULTS / "prompt_probe.json", RESULTS / "prompt_decision.json", RESULTS / "data_report.json"]},
        "model_revision": report["model"]["revision"], "model_id": cfg["model_id"],
        "human_review": review_state(), "test_generations": "not_run",
        "note": "冻结用于避免调参时改动标准，不证明人工审核完成。若需改数据，创建新版本并重新冻结。",
    }
    dump_json(FROZEN, manifest)
    print("M2 实验文件已冻结；未运行测试集，也未训练新 Adapter。")


def status():
    cfg = read_json(CONFIG)
    print("当前 M2：" + cfg["dataset_version"] + "／" + cfg["persona"])
    if (RESULTS / "data_report.json").exists():
        report = read_json(RESULTS / "data_report.json")
        if report["dataset_version"] != cfg["dataset_version"]:
            raise ValueError("数据报告过期。")
        for relative, expected in report["files"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError("数据报告的输入已变化：" + relative)
        print(f"训练 {report['training_examples']} 条／{report['training_families']} 个场景族，开发 {report['dev_scenarios']}，测试 {report['test_scenarios']}。")
    if (RESULTS / "prompt_probe.json").exists():
        probe = validate_probe(read_json(RESULTS / "prompt_probe.json"))
        print(f"当前版本开发输出 {len(probe['records'])} 条，输入证据匹配。")
    if FROZEN.exists():
        manifest = verify_frozen()
        for relative, expected in manifest["evidence_files"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError("冻结实验记录发生变化：" + relative)
        print("文件校验通过；强 Prompt：" + manifest["selected_strong_prompt"])
        print("准备包状态：已冻结；数据、Prompt、评分规则与实验记录已固定。")
    else:
        print("准备包状态：尚未冻结。")
    if (RESULTS / 'review_samples.csv').exists():
        summary = review_state()
        print(f"台词抽查：保留通过 {summary['pass']}，已改写 {summary['revisions_applied']}，新回复待复读 {summary['pending']}；改写实施与人工评分分别记录。")
        if summary['revise'] or summary['reject']:
            print('当前回复仍有需修改或拒绝的意见，需处理后再正式训练。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["dev-probe", "freeze", "status"])
    args = parser.parse_args()
    try:
        {"dev-probe": dev_probe, "freeze": freeze, "status": status}[args.step]()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"M2 步骤未通过：{error}")


if __name__ == "__main__":
    main()
