"""M1 入门实验：环境 → 下载 → 原模型 → 标签检查 → LoRA → 新进程重载。

模型和训练算法由 Transformers / PEFT / Trainer 提供。
本文件负责配置、数据接入和验证；教学解释见 docs/M1_入门教程.md。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
# 缓存放在项目内；这些设置必须早于导入 torch 和 huggingface_hub。
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CONFIG_PATH = ROOT / "configs" / "m1.json"
DATA_PATH = ROOT / "data" / "m1_smoke.jsonl"
MODEL_PATH = ROOT / "models" / "qwen2.5-1.5b-instruct"
MODEL_MANIFEST = ROOT / "models" / "model_manifest.json"
RESULTS = ROOT / "results" / "m1"
ADAPTER = ROOT / "outputs" / "m1_adapter"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors")}


def choose_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def doctor():
    import torch
    device = choose_device()
    # 不只检查标志，还真正执行一次设备上的张量计算。
    value = (torch.ones(2, device=device) + 1).cpu().tolist()
    report = {
        "recorded_at_utc": now(), "python": platform.python_version(),
        "platform": platform.system(), "architecture": platform.machine(),
        "device": device, "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(), "tensor_check": value,
        "packages": versions(),
    }
    write_json(RESULTS / "environment.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if device == "cpu":
        print("提示：只检测到 CPU。模型可以运行，但训练速度可能很慢。")


def download():
    from huggingface_hub import HfApi, snapshot_download
    cfg = read_json(CONFIG_PATH)
    if MODEL_MANIFEST.exists():
        manifest = read_json(MODEL_MANIFEST)
        if manifest["model_id"] != cfg["model_id"]:
            raise ValueError("模型配置已变化。请为新模型建立独立目录，避免覆盖旧实验。")
        revision = manifest["revision"]
    else:
        revision = HfApi().model_info(cfg["model_id"]).sha
        # 下载前先固定 commit；中断后重新运行，仍下载同一版本。
        write_json(MODEL_MANIFEST, {
            "model_id": cfg["model_id"], "revision": revision,
            "license": "apache-2.0", "created_at_utc": now(),
            "status": "downloading",
        })
    print(f"下载 {cfg['model_id']}，固定版本 {revision}", flush=True)
    snapshot_download(
        cfg["model_id"], revision=revision, local_dir=MODEL_PATH, max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE"],
    )
    if not list(MODEL_PATH.glob("*.safetensors")):
        raise RuntimeError("下载未包含模型权重，请检查网络后重新运行 download。")
    manifest = read_json(MODEL_MANIFEST)
    manifest.update(status="downloaded", completed_at_utc=now())
    write_json(MODEL_MANIFEST, manifest)
    print("模型下载完成。后面的步骤只读取本地模型，不需要 API key。")


def load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    cfg = read_json(CONFIG_PATH)
    if not MODEL_MANIFEST.exists() or read_json(MODEL_MANIFEST)["status"] != "downloaded":
        raise RuntimeError("请先运行 download，完成模型下载。")
    if read_json(MODEL_MANIFEST)["model_id"] != cfg["model_id"]:
        raise ValueError("配置中的模型与下载记录不一致。")
    set_seed(cfg["seed"])
    device = choose_device()
    dtype = torch.float32 if device == "cpu" else getattr(torch, cfg["gpu_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device)
    # 使用本地目录加载时，也保留可追溯的原始模型标识与 commit。
    model.config._name_or_path = cfg["model_id"]
    model.config._commit_hash = read_json(MODEL_MANIFEST)["revision"]
    print(f"模型已加载到 {device}，基础权重类型 {dtype}", flush=True)
    return model, tokenizer, device


def prompt_messages(cfg):
    return [{"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": cfg["probe_prompt"]}]


def prompt_tensor(tokenizer, cfg, device):
    return tokenizer.apply_chat_template(
        prompt_messages(cfg), tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


def generate(model, tokenizer, cfg, device):
    import torch
    model.eval()
    ids = prompt_tensor(tokenizer, cfg, device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise RuntimeError("原始预测分数含 NaN/Inf，先检查设备和精度，不要继续训练。")
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            temperature=1.0, top_p=1.0, top_k=50,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
    if not answer or not any(char.isalnum() for char in answer):
        raise RuntimeError("模型没有生成有效文字，请检查精度、模板和结束符。")
    return answer


def probe_logits(model, tokenizer, cfg, device):
    import torch
    model.eval()
    ids = prompt_tensor(tokenizer, cfg, device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False).logits[0, -1].float().cpu().contiguous()
    if not torch.isfinite(logits).all():
        raise RuntimeError("验证用的预测分数含 NaN/Inf，不能标记为通过。")
    return logits


def baseline():
    cfg = read_json(CONFIG_PATH)
    model, tokenizer, device = load_base()
    start = time.perf_counter()
    answer = generate(model, tokenizer, cfg, device)
    record = {
        "recorded_at_utc": now(), "model": read_json(MODEL_MANIFEST),
        "config_sha256": sha256(CONFIG_PATH), "device": device,
        "messages": prompt_messages(cfg), "answer": answer,
        "generation_seconds": round(time.perf_counter() - start, 3),
        "decoding": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"]},
    }
    write_json(RESULTS / "baseline.json", record)
    print("原模型回答：\n" + answer)


def encode_example(tokenizer, cfg, example):
    """只让目标回答参与 loss；不截断后硬凑一个没有目标的训练样本。"""
    for field in ("id", "user", "assistant"):
        if not isinstance(example.get(field), str) or not example[field].strip():
            raise ValueError(f"教学样本的 {field} 必须是非空字符串。")
    messages = [{"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": example["user"]}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["assistant"]}],
        tokenize=True, add_generation_prompt=False,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("聊天模板的回答起点不一致，不能直接使用前缀标签屏蔽。")
    if len(full_ids) > cfg["max_length"]:
        raise ValueError(f"样本 {example['id']} 超过 max_length；请先缩短样本或调整配置。")
    target_count = len(full_ids) - len(prompt_ids)
    if target_count <= 0:
        raise ValueError("样本没有可用于训练的回答 token。")
    padding = cfg["max_length"] - len(full_ids)
    return {
        "input_ids": full_ids + [tokenizer.pad_token_id] * padding,
        "attention_mask": [1] * len(full_ids) + [0] * padding,
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):] + [-100] * padding,
    }, {"id": example["id"], "prompt_tokens": len(prompt_ids),
        "answer_tokens": target_count, "padding_tokens": padding}


def encoded_data(tokenizer, cfg):
    samples = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples or len({item["id"] for item in samples}) != len(samples):
        raise ValueError("训练样本为空或 id 重复。")
    pairs = [encode_example(tokenizer, cfg, item) for item in samples]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def inspect_data():
    from transformers import AutoTokenizer
    cfg = read_json(CONFIG_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    dataset, counts = encoded_data(tokenizer, cfg)
    supervised = [token for token in dataset[0]["labels"] if token != -100]
    report = {"data_sha256": sha256(DATA_PATH), "samples": counts,
              "first_supervised_text": tokenizer.decode(supervised),
              "ignore_index": -100}
    write_json(RESULTS / "label_inspection.json", report)
    print("第一条样本真正参与训练的内容：\n" + report["first_supervised_text"])
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def train():
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import Trainer, TrainingArguments, default_data_collator

    if (ADAPTER / "adapter_model.safetensors").exists():
        raise RuntimeError("已有训练产物。请先备份或重命名 outputs/m1_adapter，避免覆盖学习记录。")
    cfg = read_json(CONFIG_PATH)
    model, tokenizer, device = load_base()
    dataset, counts = encoded_data(tokenizer, cfg)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
        bias="none", task_type="CAUSAL_LM", revision=read_json(MODEL_MANIFEST)["revision"],
    ))
    model.print_trainable_parameters()
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("可训练参数不符合只训练 LoRA 的预期。")
    print("已确认：只有 LoRA 参数参与训练。", flush=True)
    before = {name: param.detach().float().cpu().clone() for name, param in trainable.items()}
    trainable_count = sum(param.numel() for param in trainable.values())
    total_count = sum(param.numel() for param in model.parameters())
    model.config.use_cache = False
    args = TrainingArguments(
        output_dir=str(ROOT / "outputs" / "m1_trainer"),
        max_steps=cfg["max_steps"], learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=1, logging_first_step=True, save_strategy="no",
        report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, dataloader_num_workers=0,
        optim="adamw_torch", fp16=False, bf16=False,
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=default_data_collator, processing_class=tokenizer)
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("训练 loss 非有限数值，不能标记为通过。")
    changed = sum(not torch.equal(before[name], param.detach().float().cpu())
                  for name, param in trainable.items())
    if changed == 0:
        raise RuntimeError("LoRA 参数没有变化，训练未通过。")
    if any(not torch.isfinite(param).all() for param in trainable.values()):
        raise RuntimeError("LoRA 参数出现 NaN/Inf，不能保存为有效训练产物。")
    model.eval()
    adapted_logits = probe_logits(model, tokenizer, cfg, device)
    with model.disable_adapter():
        base_logits = probe_logits(model, tokenizer, cfg, device)
    effect = float((adapted_logits - base_logits).abs().max())
    if effect <= 1e-6:
        raise RuntimeError("Adapter 开关未产生可检测的 logits 差异，请检查训练。")
    answer = generate(model, tokenizer, cfg, device)
    model.save_pretrained(ADAPTER, safe_serialization=True)
    tokenizer.save_pretrained(ADAPTER)
    manifest = {
        "recorded_at_utc": now(), "process_pid": os.getpid(),
        "purpose": "M1 flow smoke test, not quality evaluation",
        "model": read_json(MODEL_MANIFEST), "device": device, "packages": versions(),
        "config": cfg, "config_sha256": sha256(CONFIG_PATH), "data_sha256": sha256(DATA_PATH),
        "sample_count": len(dataset), "token_counts": counts,
        "trainable_parameters": trainable_count, "total_parameters": total_count,
        "trainable_percent": 100 * trainable_count / total_count,
        "changed_lora_tensors": changed, "adapter_effect_max_logit_delta": effect,
        "metrics": result.metrics, "log_history": trainer.state.log_history,
        "answer_before_reload": answer,
    }
    write_json(ADAPTER / "run_manifest.json", manifest)
    write_json(RESULTS / "training.json", manifest)
    save_file({"adapted_logits": adapted_logits, "base_logits": base_logits},
              str(ADAPTER / "probe_reference.safetensors"))
    print("训练和保存完成。下一步必须另开进程执行 verify，验证真正从磁盘重载。")
    print("微调后回答（不是质量提升证据）：\n" + answer)


def verify():
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    if not (ADAPTER / "run_manifest.json").exists():
        raise RuntimeError("请先完成 train。")
    cfg = read_json(CONFIG_PATH)
    trained = read_json(ADAPTER / "run_manifest.json")
    if trained["config_sha256"] != sha256(CONFIG_PATH) or trained["data_sha256"] != sha256(DATA_PATH):
        raise ValueError("配置或教学数据已变化，不能继续认定为同一次实验。")
    if trained["model"]["revision"] != read_json(MODEL_MANIFEST)["revision"]:
        raise ValueError("基础模型版本与训练时不一致。")
    model, tokenizer, device = load_base()
    if device != trained["device"] or versions() != trained["packages"]:
        raise ValueError("设备或依赖与训练记录不同。M1 数值重载验证请使用原环境。")
    base_logits = probe_logits(model, tokenizer, cfg, device)
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False, local_files_only=True)
    adapted_logits = probe_logits(model, tokenizer, cfg, device)
    refs = load_file(str(ADAPTER / "probe_reference.safetensors"))
    base_equal = torch.allclose(base_logits, refs["base_logits"], rtol=1e-3, atol=1e-2)
    adapted_equal = torch.allclose(adapted_logits, refs["adapted_logits"], rtol=1e-3, atol=1e-2)
    effect = float((adapted_logits - base_logits).abs().max())
    answer = generate(model, tokenizer, cfg, device)
    report = {
        "recorded_at_utc": now(), "new_process_pid": os.getpid(), "device": device,
        "different_process_from_training": os.getpid() != trained["process_pid"],
        "base_logits_match": bool(base_equal), "adapter_logits_match": bool(adapted_equal),
        "reload_max_logit_error": float((adapted_logits - refs["adapted_logits"]).abs().max()),
        "adapter_effect_max_logit_delta": effect,
        "answer": answer, "answer_matches_before_reload": answer == trained["answer_before_reload"],
        "verified": bool(base_equal and adapted_equal and effect > 1e-6
                         and trained["changed_lora_tensors"] > 0
                         and os.getpid() != trained["process_pid"]),
        "config_sha256": sha256(CONFIG_PATH), "data_sha256": sha256(DATA_PATH),
        "adapter_sha256": sha256(ADAPTER / "adapter_model.safetensors"),
        "model_revision": trained["model"]["revision"],
    }
    write_json(RESULTS / "reload_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise RuntimeError("重载验证没有通过，请检查记录中的数值差异。")
    print("M1 的训练、保存、新进程重载验证已通过。此结论不代表聊天质量提升。")


def status():
    files = {"环境检查": RESULTS / "environment.json", "原模型推理": RESULTS / "baseline.json",
             "标签检查": RESULTS / "label_inspection.json", "小样本训练": RESULTS / "training.json",
             "新进程重载": RESULTS / "reload_verification.json"}
    for name, path in files.items():
        print(f"{'已有记录' if path.exists() else '待运行'}：{name}")
    valid = False
    if all(path.exists() for path in files.values()) and (ADAPTER / "adapter_model.safetensors").exists():
        verification = read_json(files["新进程重载"])
        baseline_record = read_json(files["原模型推理"])
        valid = bool(verification["verified"]
            and verification["config_sha256"] == sha256(CONFIG_PATH)
            and verification["data_sha256"] == sha256(DATA_PATH)
            and verification["adapter_sha256"] == sha256(ADAPTER / "adapter_model.safetensors")
            and baseline_record["config_sha256"] == sha256(CONFIG_PATH)
            and baseline_record["model"]["revision"] == verification["model_revision"]
            and read_json(MODEL_MANIFEST)["revision"] == verification["model_revision"])
    print("M1 状态：" + ("已验证，流程通过" if valid else "尚未完成或记录已过期"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["doctor", "download", "baseline", "inspect-data", "train", "verify", "status"])
    args = parser.parse_args()
    actions = {"doctor": doctor, "download": download, "baseline": baseline,
               "inspect-data": inspect_data, "train": train, "verify": verify, "status": status}
    try:
        actions[args.step]()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"步骤 {args.step} 未通过：{error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
