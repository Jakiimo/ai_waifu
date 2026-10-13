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
    prepare, read_json, read_jsonl, verify_frozen,
)


def measure_style(answer):
    """仅自动线索，不冒充人工评分；中文标点近似切句，不把换行当句号。"""
    sentences = [part for part in re.split(r"[。！？!?]+", answer) if part.strip()]
    return {
        "approx_sentences": len(sentences),
        "undesired_address": any(word in answer for word in ("主人", "亲爱的", "宝贝")),
        "repeated_punctuation": bool(re.search(r"[!！?？]{3,}", answer)),
        "game_metaphor_present": any(word in answer for word in ("关卡", "存档", "回血", "技能点", "副本", "经验", "地图")),
        "warning": "出现比喻不等于风格更好，句数也不等于有用性。",
    }


def dev_probe():
    import torch
    from m1 import load_base
    if FROZEN.exists():
        raise ValueError("实验已冻结。不要覆盖 Prompt 选择记录；下一轮实验请新建版本。")
    prepare()
    cfg, prompts = read_json(CONFIG), read_json(PROMPTS)
    if (RESULTS / "prompt_probe.json").exists():
        old = read_json(RESULTS / "prompt_probe.json")
        if old["config_sha256"] == digest(CONFIG) and old["prompts_sha256"] == digest(PROMPTS) and old["dev_sha256"] == digest(DEV):
            print("已有同版本开发集推理记录，未重复生成。")
            return old
        raise ValueError("已有不同配置的开发记录，请先保留旧记录，再建立新版本。")
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
        "scope": f"每类 1 题，共 {len(cfg['probe_ids'])} 个开发场景、{len(prompts)} 个 Prompt，共 {len(records)} 个回答；未运行测试集。",
        "config_sha256": digest(CONFIG), "prompts_sha256": digest(PROMPTS), "dev_sha256": digest(DEV),
        "device": device, "model": read_json(ROOT / "models" / "model_manifest.json"),
        "decoding": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"]}, "records": records,
    }
    dump_json(RESULTS / "prompt_probe.json", report)
    lines = ["# M2 开发集 Prompt 对照", "", report["scope"], "", "这些输出用于选择 Prompt，不是微调结果或最终盲评。", ""]
    for case_id in cfg["probe_ids"]:
        case = cases[case_id]
        lines.extend([f"## {case_id} {case['category']}", "", case["messages"][-1]["content"], ""])
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
    report = read_json(RESULTS / "prompt_probe.json")
    for key, path in [("config_sha256", CONFIG), ("prompts_sha256", PROMPTS), ("dev_sha256", DEV)]:
        if report[key] != digest(path):
            raise ValueError("开发实验与当前文件不一致，不能冻结。")
    decision = read_json(RESULTS / "prompt_decision.json")
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
        "human_review": "pending", "test_generations": "not_run",
        "note": "冻结用于避免调参时改动标准，不证明人工审核完成。若需改数据，创建新版本并重新冻结。",
    }
    dump_json(FROZEN, manifest)
    print("M2 实验文件已冻结；未运行测试集，也未训练新 Adapter。")


def status():
    print("M2 数据和实验设计")
    if (ROOT / "data" / "m2_tsundere_v2" / "validation_report.json").exists():
        print("当前风格已修订为傲娇 v2 候选；下方冻结信息属于旧版历史，暂不用于正式训练。")
    if (RESULTS / "data_report.json").exists():
        report = read_json(RESULTS / "data_report.json")
        print(f"训练 {report['training_examples']} 条／{report['training_families']} 个场景族，开发 {report['dev_scenarios']}，测试 {report['test_scenarios']}。")
    if FROZEN.exists():
        manifest = verify_frozen()
        for relative, expected in manifest["evidence_files"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError("冻结实验记录发生变化：" + relative)
        print("文件校验通过；强 Prompt：" + manifest["selected_strong_prompt"])
        print("准备包状态：已冻结；冻结时人工风格抽查未完成。")
        review_path = RESULTS / "review_samples.csv"
        if review_path.exists():
            with review_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            reviewed = sum(row["human_verdict"] in ("pass", "revise", "reject") for row in rows)
            flagged = sum(row["human_verdict"] in ("revise", "reject") for row in rows)
            print(f"抽查表已填写 {reviewed}/{len(rows)}，标记需处理 {flagged}；不等于全量人工审核。")
            if flagged:
                print("存在需修改或拒绝的样本：先创建修订数据版本，再进行正式训练。")
    else:
        print("准备包状态：尚未冻结。")


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
