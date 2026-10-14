"""M3：冻结数据检查 → 正式LoRA训练 → 独立进程重载 → 开发集观察。

训练与推理引擎复用Transformers/PEFT，只监督每条对话的最后一条回复。
M1和M2产物不修改，所有新结果写到独立run_id目录。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT))
from prepare_data import (FROZEN, TRAIN, DEV, PROMPTS, CONFIG as M2_CONFIG,
                          read_json, read_jsonl, dump_json, digest, verify_frozen, review_state)
from training_data import encode_conversation
from scripts.m1 import choose_device, versions, now

CONFIG = ROOT / 'configs/m3.json'
MODEL = ROOT / 'models/qwen2.5-1.5b-instruct'
MODEL_MANIFEST = ROOT / 'models/model_manifest.json'


def paths(cfg):
    run_id = cfg['run_id']
    if not isinstance(run_id,str) or not run_id.startswith('m3_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in run_id):
        raise ValueError('run_id必须以m3_开头，并只使用小写字母、数字、下划线和连字符，避免写入M1/M2目录。')
    run = ROOT / 'outputs' / run_id
    return run, run / 'adapter', ROOT / 'results' / run_id


def validate_config(cfg, m2):
    if cfg['model_id'] != m2['model_id'] or cfg['dataset_version'] != m2['dataset_version']:
        raise ValueError('M3模型或数据版本与冻结M2不同。')
    if cfg['max_length'] != m2['max_length'] or cfg['training_prompt'] != 'short':
        raise ValueError('M3使用冻结的短指令和训练长度，不静默变更M2设计。')
    if cfg['num_train_epochs'] <= 0 or cfg['batch_size'] <= 0 or cfg['gradient_accumulation_steps'] <= 0:
        raise ValueError('epoch、batch及梯度累积必须为正。')
    if not 0 < cfg['learning_rate'] < 1 or not 0 <= cfg['lora_dropout'] < 1:
        raise ValueError('学习率或dropout配置错误。')
    names = cfg['comparison_prompt_names']
    if names != ['short', m2['selected_strong_prompt']]:
        raise ValueError('开发观察只使用冻结的短/详细Prompt。')
    if 'development_observation' in cfg:
        observation = cfg['development_observation']
        ids = observation.get('case_ids', [])
        dev_ids = {case['id'] for case in read_jsonl(DEV)}
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= dev_ids:
            raise ValueError('训练中观察必须使用不重复的开发集case_ids。')
        if observation.get('prompt_name') != cfg['training_prompt']:
            raise ValueError('训练中观察使用与训练一致的short Prompt。')
        if not 0 < observation.get('max_new_tokens', 0) <= m2['max_new_tokens']:
            raise ValueError('训练中观察max_new_tokens必须为正且不超过M2生成上限。')


def preflight():
    cfg = read_json(CONFIG)
    frozen = verify_frozen()
    for relative, expected in frozen['evidence_files'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('M2冻结证据发生变化：' + relative)
    m2 = read_json(M2_CONFIG)
    validate_config(cfg, m2)
    review = review_state()
    if review['pass'] != review['sample_size'] or review['pending'] or review['revise'] or review['reject']:
        raise ValueError('当前抽查仍有未通过条目，先处理台词意见。')
    model = read_json(MODEL_MANIFEST)
    if model['status'] != 'downloaded' or model['revision'] != frozen['model_revision']:
        raise ValueError('本地模型下载或版本不符合冻结实验。')
    if model['model_id'] != cfg['model_id']:
        raise ValueError('本地基础模型标识不同。')
    return cfg, m2, frozen


def encode_rows(tokenizer, cfg, prompt, rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('训练实例为空或id重复。')
    encoded, counts = [], []
    for row in rows:
        batch, count = encode_conversation(tokenizer, prompt, row['messages'], cfg['max_length'])
        encoded.append(batch)
        counts.append({'id': row['id'], **count})
    return encoded, counts


def datasets(tokenizer, cfg):
    prompt = read_json(PROMPTS)[cfg['training_prompt']]
    rows = read_jsonl(TRAIN)
    train, counts = encode_rows(tokenizer, cfg, prompt, rows)
    dev_rows = [{'id': c['id'], 'messages': c['messages'] + [
        {'role': 'assistant', 'content': c['reference_answer']}]} for c in read_jsonl(DEV)]
    dev, _ = encode_rows(tokenizer, cfg, prompt, dev_rows)
    return train, dev, counts


def load_base(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(cfg['seed'])
    device = choose_device()
    dtype = torch.float32 if device == 'cpu' else getattr(torch, cfg['gpu_dtype'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=dtype, attn_implementation='eager', low_cpu_mem_usage=True
    ).to(device)
    model.config._name_or_path = cfg['model_id']
    model.config._commit_hash = read_json(MODEL_MANIFEST)['revision']
    print(f'基础模型：{device}／{dtype}；未加载M1 Adapter。', flush=True)
    return model, tokenizer, device


def verification_messages(cfg):
    case = next(c for c in read_jsonl(DEV) if c['id'] == cfg['verification_case_id'])
    return [{'role': 'system', 'content': read_json(PROMPTS)['short']}] + case['messages']


def logits_for(model, tokenizer, messages, device):
    import torch
    model.eval()
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors='pt').to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float().cpu().contiguous()
    if not torch.isfinite(logits).all():
        raise RuntimeError('预测分数出现NaN/Inf。')
    return logits


def generate(model, tokenizer, messages, device, m2):
    import torch
    model.eval()
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors='pt').to(device)
    if ids.shape[1] > m2['max_input_tokens']:
        raise ValueError('推理输入超过冻结上限。')
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise RuntimeError('生成前预测分数出现NaN/Inf。')
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=m2['max_new_tokens'], do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    tokens = output[0, ids.shape[1]:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not answer or not any(c.isalnum() for c in answer):
        raise RuntimeError('没有生成有效回答。')
    return {'messages': messages, 'answer': answer, 'input_tokens': ids.shape[1],
            'output_tokens': tokens.numel(),
            'finish_reason': 'eos' if int(tokens[-1]) == tokenizer.eos_token_id else 'length',
            'seconds': round(time.perf_counter() - started, 3)}


def protected_hashes():
    # 固定M1文件与M2冻结清单；不复制基础权重，不对大模型做反复全盘哈希。
    files = [ROOT / 'configs/m1.json', ROOT / 'scripts/m1.py', ROOT / 'data/m1_smoke.jsonl', FROZEN]
    files += [p for d in ['outputs/m1_adapter', 'results/m1'] for p in (ROOT / d).rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def check_provenance(record):
    if record['config_sha256'] != digest(CONFIG) or record['train_sha256'] != digest(TRAIN):
        raise ValueError('训练配置或数据已变化，请建立新的run_id。')
    if record['m2_frozen_sha256'] != digest(FROZEN):
        raise ValueError('M2冻结清单已变化。')
    if record['model'] != read_json(MODEL_MANIFEST):
        raise ValueError('基础模型记录与训练时不同。')
    for relative, expected in record['protected_file_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('受保护的M1/M2文件发生变化：' + relative)


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'RUNNING.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('此run已有训练进程，不能同时启动。')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class DevelopmentObserver:
    """固定开发问题的真实生成；不把参考答案送入推理，也不更新参数。"""

    def __init__(self, cfg, m2, tokenizer, device, results, resume_step=None):
        self.settings = cfg['development_observation']
        self.m2 = {**m2, 'max_new_tokens': self.settings['max_new_tokens']}
        self.tokenizer, self.device, self.results = tokenizer, device, results
        cases = {case['id']: case for case in read_jsonl(DEV)}
        self.cases = [cases[case_id] for case_id in self.settings['case_ids']]
        self.prompt = read_json(PROMPTS)[self.settings['prompt_name']]
        self.path = results / 'development_timeline.json'
        self.report = {'purpose': '开发过程观察，非最终测试或质量提升成绩',
                       'run_id': cfg['run_id'], 'config_sha256': digest(CONFIG),
                       'dev_sha256': digest(DEV), 'settings': self.settings,
                       'do_sample': False, 'snapshots': []}
        if self.path.exists():
            previous = read_json(self.path)
            for key in ('run_id', 'config_sha256', 'dev_sha256', 'settings'):
                if previous[key] != self.report[key]:
                    raise ValueError('开发过程观察记录与当前实验不同：' + key)
            self.report = previous
            if resume_step is not None:
                # on_evaluate先于checkpoint保存；丢弃尚未保存checkpoint的观察。
                self.report['snapshots'] = [s for s in previous['snapshots'] if s['step'] <= resume_step]
                self.report.pop('selected_checkpoint_step', None)
                self.save()

    def sample(self, model, step, epoch, loss):
        if any(snapshot['step'] == step for snapshot in self.report['snapshots']):
            return
        import torch
        import random
        import numpy as np
        # 观察结束后恢复训练状态和随机数，避免观察改变后续dropout等行为。
        modes = [(module, module.training) for module in model.modules()]
        use_cache = model.config.use_cache
        cpu_rng = torch.get_rng_state()
        mps_rng = torch.mps.get_rng_state() if self.device == 'mps' else None
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        stage = '训练前' if step == 0 else f'第{epoch:g}轮结束（更新{step}次）'
        print('\n===== 开发集观察：' + stage + ' =====', flush=True)
        records = []
        try:
            for case in self.cases:
                messages = [{'role': 'system', 'content': self.prompt}] + case['messages']
                output = generate(model, self.tokenizer, messages, self.device, self.m2)
                records.append({'case_id': case['id'], 'category': case['category'], **output})
                print(f'\n[{case["id"]}] 用户：{case["messages"][-1]["content"]}\n模型：{output["answer"]}', flush=True)
                if output['finish_reason'] == 'length':
                    print('（达到输出长度上限，这条回答可能未说完。）', flush=True)
        finally:
            for module, mode in modes:
                module.training = mode
            model.config.use_cache = use_cache
            torch.set_rng_state(cpu_rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        self.report['snapshots'].append({'stage': stage, 'step': step, 'epoch': epoch,
                                         'eval_loss': loss, 'recorded_at_utc': now(), 'records': records})
        self.save()
        print('\n开发回答已更新：' + str(self.results / 'development_timeline.md'), flush=True)

    def finalize(self, best_checkpoint):
        if best_checkpoint:
            self.report['selected_checkpoint_step'] = int(Path(best_checkpoint).name.split('-')[-1])
            self.save()

    def save(self):
        dump_json(self.path, self.report)
        lines = ['# 训练过程中：固定开发问题的回答变化', '',
                 '同一批开发消息、同一short Prompt、同一贪心解码和输出上限。这里是实际生成，参考答案没有送入推理；观察不更新参数。', '',
                 f'开发loss使用全部24条开发参考答案；这里生成的回答只覆盖选定的{len(self.cases)}条场景。两者衡量不同内容。', '']
        if 'selected_checkpoint_step' in self.report:
            lines += [f'最终选中的checkpoint：更新{self.report["selected_checkpoint_step"]}次；对应下面同一步数的回答。', '']
        for case in self.cases:
            lines += [f'## {case["id"]} · {case["category"]}', '']
            for message in case['messages']:
                role = '用户' if message['role'] == 'user' else '凛音（预置前文）'
                lines += [role + '：' + message['content'], '']
            for snapshot in self.report['snapshots']:
                row = next(r for r in snapshot['records'] if r['case_id'] == case['id'])
                lines += ['### ' + snapshot['stage'], '', row['answer'], '',
                          f'此阶段开发loss：{snapshot["eval_loss"]:.4f}；结束原因：{row["finish_reason"]}。', '']
        (self.results / 'development_timeline.md').write_text('\n'.join(lines), encoding='utf-8')


def inspect_data():
    from transformers import AutoTokenizer
    cfg, _, _ = preflight()
    _, _, results = paths(cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    train, dev, counts = datasets(tokenizer, cfg)
    rows = read_jsonl(TRAIN)
    index = next(i for i, row in enumerate(rows) if len(row['messages']) > 2)
    active = [x for x in train[index]['labels'] if x != -100]
    report = {'dataset_version': cfg['dataset_version'], 'training_examples': len(train),
              'development_examples': len(dev), 'contextual_examples': sum(len(r['messages']) > 2 for r in rows),
              'system_prompt': read_json(PROMPTS)['short'], 'config_sha256': digest(CONFIG),
              'train_sha256': digest(TRAIN), 'm2_frozen_sha256': digest(FROZEN),
              'example_id': rows[index]['id'], 'example_messages': rows[index]['messages'],
              'supervised_text': tokenizer.decode(active), 'token_counts': counts,
              'training': 'not_performed_by_this_command'}
    dump_json(results / 'data_inspection.json', report)
    print(f'训练{len(train)}条／开发{len(dev)}条／带前文{report["contextual_examples"]}条。')
    print('最后回复的监督文字：\n' + report['supervised_text'])


def train(resume=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator
    from transformers.trainer_utils import get_last_checkpoint

    cfg, m2, _ = preflight()
    run, adapter, results = paths(cfg)
    completed = run / 'run_manifest.json'
    if completed.exists():
        record = read_json(completed)
        check_provenance(record)
        for name, expected in record['adapter_files'].items():
            if digest(adapter / name) != expected:
                raise ValueError('已完成Adapter文件发生变化：' + name)
        print('此run已经训练完成，核对通过；未重复训练、未覆盖。')
        return record
    trainer_dir = run / 'trainer'
    if run.exists() and not resume:
        raise RuntimeError('已有未完成run。若有checkpoint，使用train --resume；否则更换configs/m3.json的run_id。')
    if resume and not (run / 'provenance.json').exists():
        raise RuntimeError('没有可恢复的训练来源记录。')
    with run_lock(run):
        results.mkdir(parents=True, exist_ok=True)
        if resume:
            provenance = read_json(run / 'provenance.json')
            check_provenance(provenance)
            if versions() != provenance['packages']:
                raise ValueError('恢复训练请使用原依赖环境。')
            checkpoint = get_last_checkpoint(str(trainer_dir))
            if not checkpoint:
                raise RuntimeError('尚未保存checkpoint，不能假装恢复；保留此run并更换run_id重新开始。')
        else:
            checkpoint = None
            provenance = {'created_at_utc': now(), 'model': read_json(MODEL_MANIFEST),
                'config_sha256': digest(CONFIG), 'train_sha256': digest(TRAIN),
                'm2_frozen_sha256': digest(FROZEN), 'packages': versions(),
                'protected_file_hashes': protected_hashes(), 'training_script_sha256': digest(Path(__file__))}
            dump_json(run / 'provenance.json', provenance)
            shutil.copy2(CONFIG, run / 'training_config.json')
            shutil.copy2(Path(__file__), run / 'training_script.py')
        progress = {'phase': 'loading', 'process_pid': os.getpid(), 'started_at_utc': now(),
                    'run_id': cfg['run_id'], 'resumed_from': checkpoint}
        dump_json(results / 'progress.json', progress)
        try:
            model, tokenizer, device = load_base(cfg)
            if resume and device != read_json(results / 'environment.json')['device']:
                raise ValueError('恢复训练设备与原记录不同。')
            dump_json(results / 'environment.json', {'device': device, 'weight_dtype': str(next(model.parameters()).dtype),
                                                     'packages': versions(), 'recorded_at_utc': now()})
            train_data, dev_data, counts = datasets(tokenizer, cfg)
            messages = verification_messages(cfg)
            original_base_logits = logits_for(model, tokenizer, messages, device)
            model = get_peft_model(model, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                lora_dropout=cfg['lora_dropout'], target_modules=cfg['target_modules'], bias='none',
                task_type='CAUSAL_LM', revision=read_json(MODEL_MANIFEST)['revision']))
            params = {name: param for name, param in model.named_parameters() if param.requires_grad}
            if not params or any('lora_' not in name for name in params):
                raise RuntimeError('只有LoRA参数应当可训练。')
            initial_path = run / 'initial_lora.safetensors'
            if not resume:
                save_file({name: param.detach().float().cpu().contiguous() for name, param in params.items()}, str(initial_path))
            before = load_file(str(initial_path))
            if set(before) != set(params):
                raise ValueError('恢复时的LoRA参数结构与初始记录不同。')
            trainable_count = sum(p.numel() for p in params.values())
            total_count = sum(p.numel() for p in model.parameters())
            model.print_trainable_parameters()
            print(f'正式训练：{len(train_data)}实例 × {cfg["num_train_epochs"]}轮；每次{cfg["batch_size"]}条，累计{cfg["gradient_accumulation_steps"]}次后更新。', flush=True)
            model.config.use_cache = False
            memory_samples = []
            observer = None
            if 'development_observation' in cfg:
                resume_step = int(Path(checkpoint).name.split('-')[-1]) if checkpoint else None
                observer = DevelopmentObserver(cfg, m2, tokenizer, device, results, resume_step)

            class Progress(TrainerCallback):
                def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
                    if observer is not None and state.global_step > 0:
                        observer.sample(model, state.global_step, float(state.epoch), metrics['eval_loss'])

                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step % cfg['logging_steps'] == 0 or state.global_step == 1:
                        progress.update(phase='training', global_step=state.global_step, max_steps=state.max_steps,
                                        epoch=state.epoch, updated_at_utc=now())
                        if device == 'mps':
                            sample = {'step': state.global_step, 'allocated_bytes': torch.mps.current_allocated_memory(),
                                      'driver_allocated_bytes': torch.mps.driver_allocated_memory()}
                            memory_samples.append(sample)
                        dump_json(results / 'progress.json', progress)

                def on_log(self, args, state, control, logs=None, **kwargs):
                    for name in ('loss', 'eval_loss', 'grad_norm'):
                        if name in (logs or {}) and not math.isfinite(float(logs[name])):
                            raise RuntimeError('训练或评估出现非有限数值：' + name)
                    dump_json(results / 'training_log.json', state.log_history)

                def on_save(self, args, state, control, **kwargs):
                    progress.update(last_checkpoint_step=state.global_step, updated_at_utc=now())
                    dump_json(results / 'progress.json', progress)

            args = TrainingArguments(output_dir=str(trainer_dir), num_train_epochs=cfg['num_train_epochs'],
                learning_rate=cfg['learning_rate'], per_device_train_batch_size=cfg['batch_size'],
                per_device_eval_batch_size=1, gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                lr_scheduler_type=cfg['lr_scheduler_type'], warmup_ratio=cfg['warmup_ratio'],
                weight_decay=cfg['weight_decay'], max_grad_norm=cfg['max_grad_norm'],
                logging_steps=cfg['logging_steps'], logging_first_step=True, logging_nan_inf_filter=False,
                eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
                load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
                report_to='none', seed=cfg['seed'], data_seed=cfg['seed'],
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                dataloader_pin_memory=False, dataloader_num_workers=0, optim='adamw_torch',
                fp16=False, bf16=False, remove_unused_columns=False, label_names=['labels'], disable_tqdm=True)
            # default_data_collator保留我们写好的-100标签，不重新复制input_ids作为labels。
            trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=dev_data,
                              data_collator=default_data_collator, processing_class=tokenizer, callbacks=[Progress()])
            if not resume:
                before_metrics = trainer.evaluate(metric_key_prefix='before_train')
                if not math.isfinite(before_metrics['before_train_loss']):
                    raise RuntimeError('训练前开发loss不是有限数值。')
                dump_json(results / 'before_training.json', before_metrics)
                if observer is not None:
                    observer.sample(model, 0, 0.0, before_metrics['before_train_loss'])
            else:
                before_metrics = read_json(results / 'before_training.json')
            print('训练前开发loss：' + str(before_metrics['before_train_loss']), flush=True)
            started = time.perf_counter()
            result = trainer.train(resume_from_checkpoint=checkpoint)
            elapsed = time.perf_counter() - started
            after_metrics = trainer.evaluate()
            if observer is not None:
                observer.finalize(trainer.state.best_model_checkpoint)
            if not math.isfinite(result.training_loss) or not math.isfinite(after_metrics['eval_loss']):
                raise RuntimeError('最终loss不是有限数值。')
            after_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
            changed = sum(not torch.equal(before[name], p.detach().float().cpu()) for name, p in after_params.items())
            if not changed or any(not torch.isfinite(p).all() for p in after_params.values()):
                raise RuntimeError('LoRA参数未更新或出现非有限数值。')
            model.eval()
            model.config.use_cache = True
            adapted_logits = logits_for(model, tokenizer, messages, device)
            with model.disable_adapter():
                base_logits = logits_for(model, tokenizer, messages, device)
            base_unchanged = torch.allclose(base_logits, original_base_logits, rtol=1e-3, atol=1e-2)
            effect = float((adapted_logits-base_logits).abs().max())
            if not base_unchanged or effect <= 1e-6:
                raise RuntimeError('基础预测发生异常变化，或Adapter没有可检测作用。')
            probe = generate(model, tokenizer, messages, device, m2)
            model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter)
            save_file({'base_logits': base_logits, 'adapted_logits': adapted_logits}, str(run / 'probe_reference.safetensors'))
            adapter_files = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
            record = {**provenance, 'completed_at_utc': now(), 'training_process_pid': os.getpid(),
                'purpose': 'M3 role SFT/LoRA training; final quality evaluation deferred to M5',
                'config': cfg, 'device': device, 'dtype': str(next(model.parameters()).dtype),
                'trainable_dtype': sorted({str(p.dtype) for p in after_params.values()}),
                'training_examples': len(train_data), 'development_examples': len(dev_data),
                'token_counts': counts, 'trainable_parameters': trainable_count, 'total_parameters': total_count,
                'trainable_percent': 100*trainable_count/total_count, 'changed_lora_tensors': changed,
                'base_logits_unchanged_after_training': bool(base_unchanged), 'adapter_effect_max_logit_delta': effect,
                'metrics': result.metrics, 'before_development_metrics': before_metrics,
                'after_development_metrics': after_metrics, 'best_checkpoint': trainer.state.best_model_checkpoint,
                'best_checkpoint_eval_loss': trainer.state.best_metric, 'global_step': trainer.state.global_step,
                'epoch': trainer.state.epoch, 'training_seconds_this_process': round(elapsed,3),
                'log_history': trainer.state.log_history, 'memory_samples': memory_samples,
                'memory_note': 'periodic MPS allocation snapshots, not a guaranteed peak-memory measurement',
                'verification_messages': messages, 'answer_before_reload': probe['answer'], 'probe': probe,
                'adapter_files': adapter_files, 'adapter_bytes': (adapter/'adapter_model.safetensors').stat().st_size,
                'reference_sha256': digest(run/'probe_reference.safetensors'), 'test_generations': 'not_run'}
            check_provenance(record)
            dump_json(completed, record)
            dump_json(results / 'training.json', record)
            progress.update(phase='trained_saved', global_step=trainer.state.global_step,
                            max_steps=trainer.state.max_steps, completed_at_utc=now())
            dump_json(results/'progress.json', progress)
            print('真实训练及保存完成。下一步另开进程运行verify。\n重载前模型回答：\n'+probe['answer'], flush=True)
            return record
        except BaseException as error:
            progress.update(phase='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error=str(error), updated_at_utc=now())
            dump_json(results/'progress.json', progress)
            raise


def trained_record(cfg):
    run, adapter, results = paths(cfg)
    if not (run/'run_manifest.json').exists():
        raise RuntimeError('此run尚未完成训练。')
    record = read_json(run/'run_manifest.json')
    check_provenance(record)
    if digest(run/'training_script.py') != record['training_script_sha256'] or digest(run/'training_config.json') != record['config_sha256']:
        raise ValueError('保存的训练源码或配置副本发生变化。')
    if digest(run/'probe_reference.safetensors') != record['reference_sha256']:
        raise ValueError('重载参考分数已变化。')
    for name, expected in record['adapter_files'].items():
        if digest(adapter/name) != expected:
            raise ValueError('Adapter文件已变化：'+name)
    return record, run, adapter, results


def validate_comparison_environment(record, baseline, device, dtype):
    if device != record['device'] or device != baseline['device']:
        raise ValueError('开发对照必须使用原模型记录中的同一设备。')
    if dtype != record['dtype'] or dtype != baseline['weight_dtype']:
        raise ValueError('开发对照必须使用同一基础权重精度。')
    packages = versions()
    if packages != record['packages'] or any(packages.get(k) != v for k,v in baseline['runtime'].items()):
        raise ValueError('开发对照的依赖与原模型/训练记录不同。')


def verify():
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    cfg, m2, _ = preflight()
    record, run, adapter, results = trained_record(cfg)
    if os.getpid() == record['training_process_pid']:
        raise RuntimeError('必须在训练进程之外验证重载。')
    model, tokenizer, device = load_base(cfg)
    if device != record['device'] or versions() != record['packages']:
        raise ValueError('数值重载对照使用原设备与依赖环境。')
    base = logits_for(model, tokenizer, record['verification_messages'], device)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    adapted = logits_for(model, tokenizer, record['verification_messages'], device)
    refs = load_file(str(run/'probe_reference.safetensors'))
    base_equal = torch.allclose(base, refs['base_logits'], rtol=1e-3, atol=1e-2)
    adapted_equal = torch.allclose(adapted, refs['adapted_logits'], rtol=1e-3, atol=1e-2)
    effect = float((adapted-base).abs().max())
    answer = generate(model, tokenizer, record['verification_messages'], device, m2)['answer']
    report = {'recorded_at_utc': now(), 'new_process_pid': os.getpid(), 'device': device,
              'different_process': os.getpid()!=record['training_process_pid'],
              'base_logits_match': bool(base_equal), 'adapter_logits_match': bool(adapted_equal),
              'reload_max_logit_error': float((adapted-refs['adapted_logits']).abs().max()),
              'adapter_effect_max_logit_delta': effect, 'answer': answer,
              'answer_matches_before_reload': answer==record['answer_before_reload'],
              'verified': bool(base_equal and adapted_equal and effect>1e-6 and record['changed_lora_tensors']>0),
              'adapter_sha256': digest(adapter/'adapter_model.safetensors'),
              'run_manifest_sha256': digest(run/'run_manifest.json'), 'model_revision': record['model']['revision']}
    report['verification_script_sha256'] = digest(Path(__file__))
    dump_json(results/'reload_verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['verified']:
        raise RuntimeError('新进程重载验证未通过。')
    print('M3训练、保存、新进程重载通过。效果判断仍需独立评测。')


def dev_compare():
    import torch
    from peft import PeftModel
    cfg, m2, _ = preflight()
    record, run, adapter, results = trained_record(cfg)
    verify_path = results/'reload_verification.json'
    if not verify_path.exists() or not read_json(verify_path)['verified']:
        raise RuntimeError('先完成新进程verify再观察开发回答。')
    path = results/'development_comparison.json'
    if path.exists():
        old = read_json(path)
        if old['adapter_sha256'] != digest(adapter/'adapter_model.safetensors') or old['config_sha256'] != digest(CONFIG):
            raise ValueError('已有对照属于其他Adapter或配置。')
        print('已有当前run开发观察，未重复生成。')
        return old
    baseline_report = read_json(ROOT/'results/m2/prompt_probe.json')
    base_records = baseline_report['records']
    base_by_key = {(r['prompt_name'], r['case_id']): r for r in base_records}
    cases = {c['id']:c for c in read_jsonl(DEV)}
    prompts = read_json(PROMPTS)
    model, tokenizer, device = load_base(cfg)
    validate_comparison_environment(record, baseline_report, device, str(next(model.parameters()).dtype))
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    comparisons = []
    for case_id in m2['probe_ids']:
        case = cases[case_id]
        for prompt_name in cfg['comparison_prompt_names']:
            baseline = base_by_key[(prompt_name,case_id)]
            messages = [{'role':'system','content':prompts[prompt_name]}]+case['messages']
            if messages != baseline['messages']:
                raise ValueError('原模型与LoRA的预置消息不一致。')
            adapted = generate(model, tokenizer, messages, device, m2)
            comparisons.append({'case_id':case_id,'category':case['category'],'prompt_name':prompt_name,
                                'messages':messages,'base_answer':baseline['answer'], 'adapted':adapted})
            print(f'LoRA/{prompt_name}/{case_id}：{adapted["output_tokens"]} token，{adapted["finish_reason"]}', flush=True)
    output = {'purpose':'non-blind development observation, not M5 final evaluation',
              'dataset_version':cfg['dataset_version'],'config_sha256':digest(CONFIG),
              'adapter_sha256':digest(adapter/'adapter_model.safetensors'),
              'base_probe_sha256':digest(ROOT/'results/m2/prompt_probe.json'),
              'model_revision':record['model']['revision'],'device':device,
              'weight_dtype':record['dtype'],'packages':versions(),'comparison_script_sha256':digest(Path(__file__)),
              'do_sample':False,'max_new_tokens':m2['max_new_tokens'],'test_generations':'not_run',
              'comparisons':comparisons}
    dump_json(path, output)
    lines=['# M3：原模型与正式LoRA开发观察','', '同一模型、Prompt、预置前文与贪心解码；9个开发场景×短/详细Prompt。原模型回答引用M2冻结记录，LoRA回答在此进程实际生成。非盲观察，不是最终盲评。','']
    for case_id in m2['probe_ids']:
        case=cases[case_id];lines.extend([f'## {case_id} {case["category"]}',''])
        for message in case['messages']:
            lines.extend([('用户' if message['role']=='user' else '凛音（预置前文）')+'：'+message['content'],''])
        for row in comparisons:
            if row['case_id']==case_id:
                lines.extend(['### '+row['prompt_name'],'','**原模型**','',row['base_answer'],'',
                              '**LoRA（实际生成）**','',row['adapted']['answer'],'',
                              f'结束原因：{row["adapted"]["finish_reason"]}；输出{row["adapted"]["output_tokens"]} token。',''])
    (results/'development_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


def status():
    cfg, _, _ = preflight()
    run, adapter, results = paths(cfg)
    print('当前M3 run：'+cfg['run_id'])
    timeline = results / 'development_timeline.json'
    if timeline.exists():
        snapshots = read_json(timeline)['snapshots']
        print('开发过程观察：' + ' → '.join(s['stage'] for s in snapshots))
        print('回答对照：' + str(results / 'development_timeline.md'))
    if not (run/'run_manifest.json').exists():
        if (results/'progress.json').exists():
            progress=read_json(results/'progress.json')
            print(json.dumps(progress,ensure_ascii=False,indent=2))
        else:
            print('尚未训练；可以先inspect-data，再train。')
        return
    record, _, _, _ = trained_record(cfg)
    print(f'已训练：{record["training_examples"]}实例，{record["epoch"]}轮，{record["global_step"]}次更新。')
    print(f'开发loss（同一Trainer口径）：{record["before_development_metrics"]["before_train_loss"]:.4f} → {record["after_development_metrics"]["eval_loss"]:.4f}')
    verification=results/'reload_verification.json'
    if verification.exists():
        report=read_json(verification)
        if report['adapter_sha256']!=digest(adapter/'adapter_model.safetensors') or report['run_manifest_sha256']!=digest(run/'run_manifest.json'):
            raise ValueError('重载记录与当前产物不同。')
        print('新进程重载：'+('通过' if report['verified'] else '未通过'))
    else:
        print('待运行新进程verify。')
    print('开发回答观察：'+('已保存' if (results/'development_comparison.json').exists() else '待运行dev-compare'))
    print('最终测试：未执行；开发loss下降不等于聊天质量提升。')


def main():
    global CONFIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['inspect-data','train','verify','dev-compare','status'])
    parser.add_argument('--resume',action='store_true',help='仅train：从此run最后完整checkpoint恢复')
    parser.add_argument('--config', type=Path, default=CONFIG, help='选择独立实验配置；默认configs/m3.json')
    args=parser.parse_args()
    CONFIG = args.config.resolve()
    if args.resume and args.step!='train':
        parser.error('--resume只适用于train')
    try:
        if args.step=='train':train(resume=args.resume)
        else:{'inspect-data':inspect_data,'verify':verify,'dev-compare':dev_compare,'status':status}[args.step]()
    except (RuntimeError,ValueError,OSError) as error:
        raise SystemExit('M3步骤未通过：'+str(error))


if __name__=='__main__':main()
