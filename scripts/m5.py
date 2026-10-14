"""M5：冻结60题四组生成 → 匿名评分包 → 人工评分汇总。不训练模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import demo
from prepare_data import FROZEN, PROMPTS, digest, dump_json, read_json, read_jsonl, verify_frozen
from scripts.m1 import now, versions

CONFIG = ROOT / 'configs/m5.json'
GROUPS = {
    'A': {'mode': 'base', 'prompt': 'short'},
    'B': {'mode': 'base', 'prompt': 'strong_examples'},
    'C': {'mode': 'lora', 'prompt': 'short'},
    'D': {'mode': 'lora', 'prompt': 'strong_examples'},
}
RATING_FIELDS = ['pair_id', 'reviewable', 'left_style', 'left_usefulness',
                 'right_style', 'right_usefulness', 'left_fact_error', 'right_fact_error',
                 'left_constraint_error', 'right_constraint_error', 'preference', 'notes']


def paths(cfg):
    name = cfg['evaluation_id']
    if not isinstance(name, str) or not name.startswith('m5_') or any(
            c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError('evaluation_id必须以m5_开头，只用小写字母、数字、下划线和连字符。')
    return ROOT / 'results' / name, ROOT / 'outputs' / name


def build_tasks(cases, prompts, cfg):
    tasks = []
    for case in cases:
        messages = case['messages']
        if not messages or len(messages) % 2 != 1:
            raise ValueError('测试上下文必须以用户问题结束，不能含最后的参考答案。')
        for i, message in enumerate(messages):
            if message.get('role') != ('user' if i % 2 == 0 else 'assistant') or \
               not isinstance(message.get('content'), str) or not message['content'].strip():
                raise ValueError('测试消息格式不正确：' + case['id'])
        for group, settings in cfg['groups'].items():
            # expectation只用于人工评分；不进入模型输入。
            tasks.append({'key': case['id'] + ':' + group, 'case_id': case['id'],
                          'category': case['category'], 'group': group, **settings,
                          'messages': [{'role': 'system', 'content': prompts[settings['prompt']]}] + messages})
    if len({task['key'] for task in tasks}) != len(tasks):
        raise ValueError('测试case_id重复。')
    return tasks


def preflight(config_path=CONFIG):
    cfg = read_json(config_path)
    paths(cfg)
    if cfg['groups'] != GROUPS or list(cfg['groups']) != list(GROUPS):
        raise ValueError('本轮必须固定A/B/C/D四组条件。')
    if cfg['do_sample'] is not False or cfg['expected_cases'] != 60:
        raise ValueError('正式测试固定60题和贪心解码，不能只挑有利题目。')
    if not isinstance(cfg['pairs_per_batch'], int) or cfg['pairs_per_batch'] <= 0:
        raise ValueError('pairs_per_batch必须为正整数。')
    if not isinstance(cfg['blind_seed'], int):
        raise ValueError('blind_seed必须为整数。')
    frozen = verify_frozen()
    test_path, rubric_path = ROOT / cfg['test_file'], ROOT / cfg['rubric_file']
    if str(test_path.relative_to(ROOT)) not in frozen['files'] or \
       digest(test_path) != frozen['files'][str(test_path.relative_to(ROOT))]:
        raise ValueError('必须使用M2冻结的最终测试集。')
    if str(rubric_path.relative_to(ROOT)) not in frozen['files'] or \
       digest(rubric_path) != frozen['files'][str(rubric_path.relative_to(ROOT))]:
        raise ValueError('必须使用M2冻结的评分规则。')
    rubric = read_json(rubric_path)
    cases = read_jsonl(test_path)
    if len(cases) != cfg['expected_cases'] or rubric['version'] != cfg['rubric_version']:
        raise ValueError('测试数量或评分版本不同。')
    if any(c['dataset_version'] != cfg['dataset_version'] or c['rubric_version'] != cfg['rubric_version'] for c in cases):
        raise ValueError('测试场景版本不同。')
    expected_categories = {k: 10 for k in ['daily', 'comfort', 'study', 'technical', 'stability', 'relationship']}
    if dict(Counter(c['category'] for c in cases)) != expected_categories:
        raise ValueError('测试类别覆盖与冻结设计不同。')
    demo_path = ROOT / cfg['demo_config']
    demo_cfg, train_cfg, adapter, prompts = demo.validate_assets(demo_path)
    for key in ('max_input_tokens', 'max_new_tokens', 'do_sample'):
        if cfg[key] != demo_cfg[key]:
            raise ValueError('正式评测与已验证推理设置不同：' + key)
    training_record = read_json(adapter.parent / 'run_manifest.json')
    files = [config_path, demo_path, ROOT / demo_cfg['training_config'], test_path, rubric_path,
             PROMPTS, FROZEN, ROOT / 'docs/凛音_角色卡.md', ROOT / 'models/model_manifest.json',
             adapter / 'adapter_model.safetensors', adapter.parent / 'run_manifest.json',
             ROOT / 'demo.py', Path(__file__), ROOT / 'scripts/m5_review.py', ROOT / 'requirements-m4-lock.txt']
    conditions = {'evaluation_id': cfg['evaluation_id'], 'dataset_version': cfg['dataset_version'],
                  'rubric_version': cfg['rubric_version'], 'training_run_id': train_cfg['run_id'],
                  'model_id': training_record['model']['model_id'], 'model_revision': training_record['model']['revision'],
                  'expected_device': training_record['device'], 'expected_dtype': training_record['dtype'],
                  'packages': versions(), 'groups': cfg['groups'],
                  'generation': {k: cfg[k] for k in ('max_input_tokens', 'max_new_tokens', 'do_sample')},
                  'temperature': 1.0, 'top_p': 1.0, 'top_k': 50,
                  'files': {str(Path(p).relative_to(ROOT)): digest(p) for p in files}}
    return cfg, cases, rubric, build_tasks(cases, prompts, cfg), conditions


def prepare(config_path=CONFIG):
    cfg, cases, rubric, tasks, conditions = preflight(config_path)
    folder, private = paths(cfg)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['conditions'] != conditions:
            raise ValueError('本轮条件已变化，请恢复原文件或另建evaluation_id；不覆盖已有评测。')
        if not (private / 'source_snapshot/scripts/m5.py').exists() or \
           digest(private / 'source_snapshot/scripts/m5.py') != conditions['files']['scripts/m5.py']:
            raise ValueError('评测源码快照缺失或已变化。')
        print('已有评测准备包，条件核对通过。', flush=True)
        return cfg, cases, rubric, tasks, manifest
    if (folder / 'generations.jsonl').exists():
        raise ValueError('已有回答但缺少manifest，不将来源不明的回答并入本轮。')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen2.5-1.5b-instruct', local_files_only=True)
    counts = []
    for task in tasks:
        count = len(tokenizer.apply_chat_template(task['messages'], tokenize=True, add_generation_prompt=True))
        if count > cfg['max_input_tokens']:
            raise ValueError('测试输入超过上限，不能截断后宣称同题比较：' + task['key'])
        counts.append({'key': task['key'], 'input_tokens': count})
    folder.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    for relative in ['scripts/m5.py', 'scripts/m5_review.py', 'demo.py', str(config_path.relative_to(ROOT)),
                     cfg['demo_config'], cfg['test_file'], cfg['rubric_file'], 'configs/m2_prompts.json']:
        target = private / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    manifest = {'prepared_at_utc': now(), 'conditions': conditions, 'test_cases': len(cases),
                'expected_generations': len(tasks), 'category_counts': dict(Counter(c['category'] for c in cases)),
                'contextual_cases': sum(len(c['messages']) > 1 for c in cases),
                'input_token_counts': counts, 'generation_at_prepare': 'not_run',
                'grading_plan': 'B/D主对照60对，A/C辅对照60对；四组各60回答评分，单作者局限需明示。'}
    dump_json(manifest_path, manifest)
    print(f'准备通过：{len(cases)}题 × 4组 = {len(tasks)}条；没有生成测试回答。', flush=True)
    return cfg, cases, rubric, tasks, manifest


def read_generations(folder, tasks):
    path = folder / 'generations.jsonl'
    if not path.exists():
        return []
    expected = {task['key']: task for task in tasks}
    records, seen = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'生成记录第{line_number}行不完整；保留文件，不能悄悄跳过。')
        key = row.get('key')
        if key not in expected or key in seen:
            raise ValueError('生成记录含重复或未知题目：' + str(key))
        task = expected[key]
        for field in ('case_id', 'group', 'mode', 'prompt', 'messages', 'category'):
            if row.get(field) != task[field]:
                raise ValueError('回答条件与固定任务不同：' + key)
        if row.get('status') not in ('ok', 'error'):
            raise ValueError('未知生成状态：' + key)
        if row['status'] == 'ok' and (not row.get('answer', '').strip() or row.get('finish_reason') not in ('eos', 'length')):
            raise ValueError('成功回答缺少文本或结束原因：' + key)
        seen.add(key)
        records.append(row)
    return records


def generate_records(folder, cfg, tasks, engine, resume=False):
    """逐条保存；测试框架可以注入假引擎，开发检查不产生正式测试回答。"""
    records = read_generations(folder, tasks)
    if len(records) == len(tasks):
        print('四组回答已齐备，未重复生成。', flush=True)
        return records
    if records and not resume:
        raise ValueError('已有未完成回答，请使用generate --resume继续；不覆盖旧回答。')
    engine.ensure_loaded()
    seen = {row['key'] for row in records}
    for task in tasks:
        if task['key'] in seen:
            continue
        try:
            # 每条用固定messages，不借用上一题或任何一组的生成回复。
            with engine.lock:
                result = engine._generate(task['messages'], task['mode'])
            row = {**task, **result, 'status': 'ok', 'recorded_at_utc': now()}
        except KeyboardInterrupt:
            dump_json(folder / 'progress.json', {'phase': 'interrupted', 'saved': len(records),
                                                'total': len(tasks), 'updated_at_utc': now()})
            raise
        except Exception as error:
            row = {**task, 'status': 'error', 'answer': '', 'finish_reason': 'error',
                   'error_type': type(error).__name__, 'error': str(error), 'recorded_at_utc': now(),
                   'input_tokens': None, 'output_tokens': 0, 'seconds': None}
        with (folder / 'generations.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        records.append(row)
        seen.add(task['key'])
        dump_json(folder / 'progress.json', {'phase': 'generating', 'saved': len(records),
                                            'total': len(tasks), 'last_key': task['key'],
                                            'errors': sum(r['status'] == 'error' for r in records), 'updated_at_utc': now()})
        print(f'[{len(records)}/{len(tasks)}] {task["key"]}：{row["status"]}，'
              f'{row["output_tokens"]} token／{row["finish_reason"]}', flush=True)
    return records


def generate(config_path=CONFIG, resume=False):
    cfg, cases, rubric, tasks, manifest = prepare(config_path)
    folder, private = paths(cfg)
    with (private / 'RUNNING.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('本轮已有生成进程，不能同时启动。')
        records = read_generations(folder, tasks)
        if len(records) < len(tasks):
            if records and not resume:
                raise ValueError('已有未完成回答，请使用generate --resume。')
            engine = demo.ChatEngine(ROOT / cfg['demo_config'])
            engine.ensure_loaded()
            if engine.device != manifest['conditions']['expected_device'] or \
               str(next(engine.model.parameters()).dtype) != manifest['conditions']['expected_dtype']:
                raise ValueError('正式评测应使用与训练验证相同的设备和基础权重精度。')
            # 接入检查只用M3已有探针，不能用最终测试回答来调方案。
            import torch
            from safetensors.torch import load_file
            from scripts.m3 import logits_for
            train_record = read_json(engine.adapter.parent / 'run_manifest.json')
            refs = load_file(str(engine.adapter.parent / 'probe_reference.safetensors'))
            with engine.model.disable_adapter():
                base_logits = logits_for(engine.model, engine.tokenizer, train_record['verification_messages'], engine.device)
            engine.model.requires_grad_(False)
            adapted_logits = logits_for(engine.model, engine.tokenizer, train_record['verification_messages'], engine.device)
            if not torch.allclose(base_logits, refs['base_logits'], rtol=1e-3, atol=1e-2) or \
               not torch.allclose(adapted_logits, refs['adapted_logits'], rtol=1e-3, atol=1e-2):
                raise ValueError('评测引擎与M3保存的预测分数不同，停止正式生成。')
            records = generate_records(folder, cfg, tasks, engine, resume)
            if any(p.requires_grad for p in engine.model.parameters()):
                raise ValueError('评测后参数未保持冻结。')
        # 再核对产物与冻结数据：没有训练，没有改Prompt或试题。
        if preflight(config_path)[4] != manifest['conditions']:
            raise ValueError('生成中实验条件发生变化，保留记录并停止交付。')
        completed = {'completed_at_utc': now(), 'records': len(records),
                     'success': sum(r['status'] == 'ok' for r in records),
                     'errors': sum(r['status'] == 'error' for r in records),
                     'generations_sha256': digest(folder / 'generations.jsonl'),
                     'manifest_sha256': digest(folder / 'manifest.json'), 'training_executed': False}
        path = folder / 'generation_complete.json'
        if path.exists():
            old = read_json(path)
            if old['generations_sha256'] != completed['generations_sha256'] or old['manifest_sha256'] != completed['manifest_sha256']:
                raise ValueError('已完成的原始生成记录已变化。')
        else:
            dump_json(path, completed)
        dump_json(folder / 'progress.json', {'phase': 'generated', 'saved': len(records), 'total': len(tasks),
                                            'errors': completed['errors'], 'updated_at_utc': now()})
        print('四组原始回答已保存。下一步blind生成匿名评分包。', flush=True)


def build_pairs(records, cases, seed):
    by_key = {row['key']: row for row in records}
    rng = random.Random(seed)
    pairs = []
    for name, original, adapted in [('BD', 'B', 'D'), ('AC', 'A', 'C')]:
        # 每种对照LoRA恰好一半在左，一半在右，再随机打乱。
        adapted_left = [True] * (len(cases) // 2) + [False] * (len(cases) - len(cases) // 2)
        rng.shuffle(adapted_left)
        for case, on_left in zip(cases, adapted_left):
            left_group, right_group = (adapted, original) if on_left else (original, adapted)
            left, right = by_key[case['id'] + ':' + left_group], by_key[case['id'] + ':' + right_group]
            pairs.append({'case_id': case['id'], 'comparison': name,
                          'left_key': left['key'], 'right_key': right['key'],
                          'left_group': left_group, 'right_group': right_group,
                          'reviewable': left['status'] == 'ok' and right['status'] == 'ok'})
    rng.shuffle(pairs)
    for i, pair in enumerate(pairs, 1):
        pair['pair_id'] = f'pair-{i:03}'
    return pairs


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def blind(config_path=CONFIG):
    cfg, cases, rubric, tasks, manifest = prepare(config_path)
    folder, private = paths(cfg)
    completed = read_json(folder / 'generation_complete.json')
    if completed['generations_sha256'] != digest(folder / 'generations.jsonl') or \
       completed['manifest_sha256'] != digest(folder / 'manifest.json'):
        raise ValueError('原始回答或manifest已变化，不能混用匿名包。')
    records = read_generations(folder, tasks)
    if len(records) != len(tasks):
        raise ValueError('四组回答尚未齐备。')
    review = folder / 'review'
    if (review / 'integrity.json').exists():
        check_blind_integrity(folder, private)
        print('匿名包已存在，未覆盖你的评分。', flush=True)
        return
    if review.exists():
        raise ValueError('已有未完成的匿名包，保留文件，不覆盖已有评分。')
    review.mkdir()
    pairs = build_pairs(records, cases, cfg['blind_seed'])
    by_key = {row['key']: row for row in records}
    case_by_id = {case['id']: case for case in cases}
    # 评分网页只读取匿名材料，不解析私有身份映射。
    items = []
    for pair in pairs:
        case = case_by_id[pair['case_id']]
        items.append({'pair_id': pair['pair_id'], 'reviewable': pair['reviewable'],
                      'category': case['category'], 'messages': case['messages'],
                      'expectation': case['expectation'],
                      'left_answer': by_key[pair['left_key']]['answer'] if by_key[pair['left_key']]['status'] == 'ok' else '〔本次生成失败〕',
                      'right_answer': by_key[pair['right_key']]['answer'] if by_key[pair['right_key']]['status'] == 'ok' else '〔本次生成失败〕'})
    dump_json(review / 'review_items.json', items)
    dump_json(private / 'blind_key.json', {'seed': cfg['blind_seed'], 'pairs': pairs})
    ratings = [{**dict.fromkeys(RATING_FIELDS, ''), 'pair_id': pair['pair_id'],
                'reviewable': 'yes' if pair['reviewable'] else 'no'} for pair in pairs]
    write_csv(review / 'ratings_template.csv', ratings)
    shutil.copy2(review / 'ratings_template.csv', review / 'ratings.csv')
    guide = ['# M5 匿名评分说明', '', '左右身份已隐藏；不要先看原始回答或身份映射。仅凭语气猜测身份仍可能产生偏差。', '',
             '每行对应一对回答，先读batch文件，再填写ratings.csv。风格与有用性各1–5；事实错误和违反要求填yes/no/na；偏好填left/tie/right。', '',
             '技术类事实错误必须填yes/no；其他没有事实可核查的情形可以填na。没有明确要求可检查时，违反要求可以填na。na不算“正确”，汇总会单独列出。', '',
             '偏好先看准确、有用、是否遵守要求，再看角色风格；可以平局。不要按口癖、姐姐称呼或波浪号计分。', '',
             'reviewable=no表示至少一边生成失败；保留在表里，不填写分数或偏好。失败仍进入自动统计，不当作平局。', '',
             '可以分批保存。只保存评分内容，不修改pair_id、reviewable或新增/删掉行。只有一位作者评分时，结论须说明这一局限。', '',
             '## 角色设定', '', (ROOT / 'docs/凛音_角色卡.md').read_text(), '', '## 评分锚点', '']
    for title, key in [('角色风格', 'persona_consistency'), ('有用性', 'usefulness')]:
        guide += ['### ' + title, ''] + [f'{score}：{text}' for score, text in rubric[key].items()] + ['']
    (review / '评分说明.md').write_text('\n'.join(guide), encoding='utf-8')
    batch_files = []
    for start in range(0, len(pairs), cfg['pairs_per_batch']):
        batch = start // cfg['pairs_per_batch'] + 1
        path = review / f'batch_{batch:02}.md'
        lines = [f'# 匿名评分第{batch}批', '', '只显示当前题的对话前文与左右回答，身份和实际system Prompt均隐藏。', '']
        for pair in pairs[start:start + cfg['pairs_per_batch']]:
            case = case_by_id[pair['case_id']]
            lines += ['## ' + pair['pair_id'], '', '类别：' + case['category'], '']
            for message in case['messages']:
                label = '用户' if message['role'] == 'user' else '预置角色前文'
                lines += [label + '：' + message['content'], '']
            lines += ['检查重点：' + case['expectation'], '', '### 左侧回答', '',
                      by_key[pair['left_key']]['answer'] if by_key[pair['left_key']]['status'] == 'ok' else '〔本次生成失败〕', '',
                      '### 右侧回答', '',
                      by_key[pair['right_key']]['answer'] if by_key[pair['right_key']]['status'] == 'ok' else '〔本次生成失败〕', '']
            if not pair['reviewable']:
                lines += ['本对不可评分，保留失败记录。', '']
        path.write_text('\n'.join(lines), encoding='utf-8')
        batch_files.append(path)
    immutable = [private / 'blind_key.json', review / 'ratings_template.csv', review / '评分说明.md', review / 'review_items.json'] + batch_files
    dump_json(review / 'integrity.json', {'generations_sha256': completed['generations_sha256'],
              'manifest_sha256': completed['manifest_sha256'], 'pairs': len(pairs),
              'files': {str(p.relative_to(ROOT)): digest(p) for p in immutable},
              'editable_file': str((review / 'ratings.csv').relative_to(ROOT))})
    print(f'匿名包已生成：{len(pairs)}对／{len(batch_files)}批。填写：{review / "ratings.csv"}', flush=True)


def check_blind_integrity(folder, private):
    seal = read_json(folder / 'review/integrity.json')
    if seal['generations_sha256'] != digest(folder / 'generations.jsonl') or \
       seal['manifest_sha256'] != digest(folder / 'manifest.json'):
        raise ValueError('匿名包与本轮原始回答不同。')
    for relative, expected in seal['files'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('匿名材料或身份映射已变化：' + relative)
    return seal


def parse_ratings(path, pairs, cases):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RATING_FIELDS:
            raise ValueError('评分CSV列名或顺序已变化，请使用本轮模板。')
        rows = list(reader)
    expected = {pair['pair_id']: pair for pair in pairs}
    categories = {case['id']: case['category'] for case in cases}
    if len(rows) != len(pairs) or {row['pair_id'] for row in rows} != set(expected):
        raise ValueError('评分CSV含缺失、重复或未知pair_id。')
    parsed, pending = [], []
    fill_fields = [field for field in RATING_FIELDS if field not in ('pair_id', 'reviewable', 'notes')]
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('CSV字段数量不正确，检查含逗号的备注是否正确加引号。')
        pair = expected[row['pair_id']]
        reviewable = 'yes' if pair['reviewable'] else 'no'
        if row['reviewable'].strip().lower() != reviewable:
            raise ValueError('不可修改reviewable：' + pair['pair_id'])
        fields = {key: row[key].strip().lower() for key in fill_fields}
        if not pair['reviewable']:
            if any(fields.values()):
                raise ValueError('生成失败的对照不能填写人为分数或平局：' + pair['pair_id'])
            continue
        # 部分已填也检查非法值；不能把错填当作尚未评分静默跳过。
        for key, value in fields.items():
            allowed = {'1', '2', '3', '4', '5'} if key.endswith(('style', 'usefulness')) else \
                      {'left', 'tie', 'right'} if key == 'preference' else {'yes', 'no', 'na'}
            if value and value not in allowed:
                raise ValueError(f'{pair["pair_id"]}的{key}无效：{value}')
        if any(not value for value in fields.values()):
            pending.append(pair['pair_id'])
            continue
        if categories[pair['case_id']] == 'technical' and any(fields[side + '_fact_error'] == 'na' for side in ('left', 'right')):
            raise ValueError('技术题事实错误必须明确填yes/no：' + pair['pair_id'])
        parsed.append({**pair, **fields, 'notes': row['notes'],
                       **{key: int(value) for key, value in fields.items() if key.endswith(('style', 'usefulness'))}})
    return parsed, pending


def mean(values):
    return round(statistics.mean(values), 4) if values else None


def automatic_metrics(records, rubric, input_counts=None):
    output = {}
    for group in GROUPS:
        rows = [row for row in records if row['group'] == group]
        good = [row for row in rows if row['status'] == 'ok']
        daily = [row for row in good if row['category'] == 'daily']
        low, high = rubric['automatic_checks']['daily_sentence_range']
        counts = [len([s for s in re.split(r'[。！？!?]+', row['answer']) if s.strip()]) for row in daily]
        input_values = [input_counts[r['key']] for r in rows] if input_counts is not None else \
                       [r['input_tokens'] for r in good if r['input_tokens'] is not None]
        output[group] = {'planned': len(rows), 'generated': len(good), 'errors': len(rows) - len(good),
                         'truncated': sum(row['finish_reason'] == 'length' for row in good),
                         'mean_input_tokens': mean(input_values), 'input_token_denominator': len(input_values),
                         'mean_output_tokens': mean([r['output_tokens'] for r in good]),
                         'exact_duplicate_outputs': len(good) - len({r['answer'] for r in good}),
                         'undesired_address_outputs': sum(any(term in r['answer'] for term in rubric['automatic_checks']['undesired_addresses']) for r in good),
                         'daily_sentence_rule_pass': sum(low <= n <= high for n in counts),
                         'daily_sentence_rule_denominator': len(daily),
                         'rule_note': '按标点切句的启发式，仅日常类别；不能当作角色质量评分。'}
    return output


def summarize(records, pairs, ratings, cases, rubric, input_counts=None):
    automatic = automatic_metrics(records, rubric, input_counts)
    scored = {group: [] for group in GROUPS}
    preferences = {}
    for row in ratings:
        for side in ('left', 'right'):
            scored[row[side + '_group']].append({'case_id': row['case_id'],
                'style': row[side + '_style'], 'usefulness': row[side + '_usefulness'],
                'fact_error': row[side + '_fact_error'], 'constraint_error': row[side + '_constraint_error']})
    human = {}
    for group, rows in scored.items():
        human[group] = {'rated_answers': len(rows), 'mean_style': mean([r['style'] for r in rows]),
                        'mean_usefulness': mean([r['usefulness'] for r in rows]),
                        'fact_error_yes': sum(r['fact_error'] == 'yes' for r in rows),
                        'fact_error_marked_denominator': sum(r['fact_error'] in ('yes', 'no') for r in rows),
                        'fact_error_na': sum(r['fact_error'] == 'na' for r in rows),
                        'constraint_error_yes': sum(r['constraint_error'] == 'yes' for r in rows),
                        'constraint_error_marked_denominator': sum(r['constraint_error'] in ('yes', 'no') for r in rows),
                        'constraint_error_na': sum(r['constraint_error'] == 'na' for r in rows)}
    for comparison, adapted_group in [('BD', 'D'), ('AC', 'C')]:
        rows = [row for row in ratings if row['comparison'] == comparison]
        win = tie = loss = 0
        for row in rows:
            if row['preference'] == 'tie':
                tie += 1
            elif row[row['preference'] + '_group'] == adapted_group:
                win += 1
            else:
                loss += 1
        candidates = [pair for pair in pairs if pair['comparison'] == comparison]
        valid = sum(pair['reviewable'] for pair in candidates)
        preferences[comparison] = {'planned_pairs': len(candidates), 'generation_failed_pairs': len(candidates) - valid,
                                   'reviewable_pairs': valid, 'rated_pairs': len(rows),
                                   'pending_pairs': valid - len(rows), 'lora_wins': win, 'ties': tie, 'lora_losses': loss,
                                   'win_rate_all_rated': round(win / len(rows), 4) if rows else None,
                                   'win_rate_excluding_ties': round(win / (win + loss), 4) if win + loss else None,
                                   'denominator_note': '主胜率=LoRA胜/全部已完整评分的有效配对；排除平局另列，生成失败及未评分数量单列。'}
    # C与B是不同Prompt的整套方案；只报告同题评分差，不冒充成对偏好票。
    c_by_id, b_by_id = ({r['case_id']: r for r in scored[group]} for group in ('C', 'B'))
    common = sorted(set(c_by_id) & set(b_by_id))
    secondary = {'paired_scored_cases': len(common),
                 'mean_C_minus_B_style': mean([c_by_id[k]['style'] - b_by_id[k]['style'] for k in common]),
                 'mean_C_minus_B_usefulness': mean([c_by_id[k]['usefulness'] - b_by_id[k]['usefulness'] for k in common]),
                 'note': '同题评分差，不是C/B直接偏好票；来自不同匿名对照，评分语境可能影响数值，且Prompt不同。'}
    categories = {}
    for category in sorted({case['category'] for case in cases}):
        ids = {case['id'] for case in cases if case['category'] == category}
        categories[category] = {group: {'rated': sum(r['case_id'] in ids for r in rows),
            'mean_style': mean([r['style'] for r in rows if r['case_id'] in ids]),
            'mean_usefulness': mean([r['usefulness'] for r in rows if r['case_id'] in ids])}
            for group, rows in scored.items()}
    return {'automatic': automatic, 'human': human, 'pairwise': preferences,
            'secondary_C_vs_B': secondary, 'category_scores': categories}


def report(config_path=CONFIG, reviewer=None, allow_partial=False):
    if not reviewer or not reviewer.strip():
        raise ValueError('请用--reviewer记录实际评分者昵称或“作者本人”。')
    cfg, cases, rubric, tasks, manifest = prepare(config_path)
    folder, private = paths(cfg)
    check_blind_integrity(folder, private)
    pairs = read_json(private / 'blind_key.json')['pairs']
    ratings, pending = parse_ratings(folder / 'review/ratings.csv', pairs, cases)
    if pending and not allow_partial:
        raise ValueError(f'还有{len(pending)}对未完整评分，例如{pending[:3]}。继续填写，或用--allow-partial看未完成预览。')
    records = read_generations(folder, tasks)
    input_counts = {row['key']: row['input_tokens'] for row in manifest['input_token_counts']}
    summary = summarize(records, pairs, ratings, cases, rubric, input_counts)
    successful = [{row['case_id'] for row in records if row['group'] == group and row['status'] == 'ok'} for group in GROUPS]
    common_success = set.intersection(*successful)
    sufficient = len(common_success) >= 40 and summary['pairwise']['BD']['rated_pairs'] >= 40
    ratings_hash = digest(folder / 'review/ratings.csv')
    snapshots = folder / 'ratings_snapshots'
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / (ratings_hash + '.csv')
    if snapshot.exists() and digest(snapshot) != ratings_hash:
        raise ValueError('已有评分快照已变化，停止覆盖。')
    if not snapshot.exists():
        shutil.copy2(folder / 'review/ratings.csv', snapshot)
    summary.update(recorded_at_utc=now(), evaluation_id=cfg['evaluation_id'],
                   reviewer=reviewer.strip(), reviewer_count=1, complete=not pending and sufficient,
                   ratings_complete=not pending, quality_comparison_usable=sufficient,
                   four_group_successful_cases=len(common_success), minimum_valid_cases=40,
                   ratings_sha256=ratings_hash, ratings_snapshot=str(snapshot.relative_to(ROOT)), pending_pair_ids=pending,
                   conditions=manifest['conditions'],
                   limitations=['只有一位评分者；可能熟悉数据或从语气猜出模型身份。',
                                '小型合成测试集，主题与训练集可能相近；不是全量语义泄漏证明。',
                                '一次贪心生成与描述性人工评分，不宣称统计显著或普遍改善。',
                                '测试已用于这次评测，后续改模型/Prompt应另建最终测试集。',
                                '四组人设分数均值只是1–5主观评分摘要，输入token只表示输入长度，不自动等于费用或速度收益。'])
    suffix = '_partial' if pending else ''
    dump_json(folder / f'summary{suffix}.json', summary)
    lines = ['# M5 评测报告' + ('（未完成预览）' if pending else '（有效样本不足）' if not sufficient else ''), '',
             f'评分者：{reviewer.strip()}；评分者数量：1；完整评分{len(ratings)}对，待评分{len(pending)}对。', '',
             f'四组都成功生成的独立场景：{len(common_success)}/60；至少40题及40对完整BD评分才满足本轮质量比较门槛。', '',
             'A=原模型+短Prompt；B=原模型+详细Prompt；C=LoRA+短Prompt；D=LoRA+详细Prompt。', '',
             '| 组别 | 成功/计划 | 截断 | 全部固定试题输入token均值 | 已评分回答 | 风格均分 | 有用性均分 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        auto, human = summary['automatic'][group], summary['human'][group]
        lines.append(f'| {group} | {auto["generated"]}/{auto["planned"]} | {auto["truncated"]} | '
                     f'{auto["mean_input_tokens"]} | {human["rated_answers"]} | {human["mean_style"]} | {human["mean_usefulness"]} |')
    lines += ['', '## 成对偏好', '', '| 对照 | LoRA胜 | 平 | LoRA负 | 已评分分母 | 主胜率 | 生成失败对 | 未评分对 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['pairwise'].items():
        lines.append(f'| {name} | {row["lora_wins"]} | {row["ties"]} | {row["lora_losses"]} | '
                     f'{row["rated_pairs"]} | {row["win_rate_all_rated"]} | {row["generation_failed_pairs"]} | {row["pending_pairs"]} |')
    lines += ['', '主胜率=LoRA胜/全部完整评分的有效配对。平局计入分母；失败与未评分单列。排除平局的胜率见JSON。', '',
              '## 错误标记', '', '| 组别 | 事实错误yes/明确检查数 | 事实na | 违反要求yes/明确检查数 | 违反要求na |',
              '|---|---:|---:|---:|---:|']
    for group, row in summary['human'].items():
        lines.append(f'| {group} | {row["fact_error_yes"]}/{row["fact_error_marked_denominator"]} | '
                     f'{row["fact_error_na"]} | {row["constraint_error_yes"]}/{row["constraint_error_marked_denominator"]} | {row["constraint_error_na"]} |')
    secondary = summary['secondary_C_vs_B']
    lines += ['', '## C与B的整套方案对比', '',
              f'同题已评分{secondary["paired_scored_cases"]}题；C−B风格均分差{secondary["mean_C_minus_B_style"]}，'
              f'有用性均分差{secondary["mean_C_minus_B_usefulness"]}。', '', secondary['note'], '',
              '## 代表案例与失败', '']
    by_key = {row['key']: row for row in records}
    selected = []
    primary = [row for row in ratings if row['comparison'] == 'BD']
    for label in ('lora_win', 'tie', 'lora_loss'):
        matches = [row for row in primary if ('tie' if row['preference'] == 'tie' else
                   'lora_win' if row[row['preference'] + '_group'] == 'D' else 'lora_loss') == label]
        selected.extend(matches[:2])
    for row in selected:
        case = next(case for case in cases if case['id'] == row['case_id'])
        lines += ['### ' + row['case_id'], '', '用户：' + case['messages'][-1]['content'], '',
                  '**B 原模型+详细Prompt**', '', by_key[row['case_id'] + ':B']['answer'], '',
                  '**D LoRA+详细Prompt**', '', by_key[row['case_id'] + ':D']['answer'], '', '评分者备注：' + row['notes'], '']
    failed = [row for row in records if row['status'] == 'error']
    lines += [f'生成失败共{len(failed)}条，原始错误保留在generations.jsonl。']
    for row in failed[:10]:
        lines += ['- ' + row['key'] + '：' + row['error_type'] + ' / ' + row['error']]
    lines += ['', '## 如何判断', '',
              '先看主对照BD的偏好、风格、有用性与错误，不用风格分升高掩盖事实或约束退化。结果只说明本轮样本与评分条件下的表现。最终技术取舍需要结合实际案例，由作者写出，不自动生成“显著提升”结论。', '',
              '## 局限', ''] + ['- ' + note for note in summary['limitations']]
    (folder / f'report{suffix}.md').write_text('\n'.join(lines), encoding='utf-8')
    print('已生成' + ('未完成预览' if pending else '完整评分报告' if sufficient else '有效样本不足的诊断报告') + '：' + str(folder / f'report{suffix}.md'), flush=True)


def status(config_path=CONFIG):
    cfg, cases, rubric, tasks, conditions = preflight(config_path)
    folder, private = paths(cfg)
    records = read_generations(folder, tasks)
    print('M5实验：' + cfg['evaluation_id'])
    print(f'原始回答：{len(records)}/{len(tasks)}；失败{sum(r["status"]=="error" for r in records)}。')
    if not (folder / 'manifest.json').exists():
        print('尚未prepare，未生成最终测试回答。')
        return
    if read_json(folder / 'manifest.json')['conditions'] != conditions:
        raise ValueError('实验条件与准备时不同。')
    if (folder / 'review/integrity.json').exists():
        check_blind_integrity(folder, private)
        pairs = read_json(private / 'blind_key.json')['pairs']
        rated, pending = parse_ratings(folder / 'review/ratings.csv', pairs, cases)
        print(f'匿名评分：完整{len(rated)}对，待评{len(pending)}对，生成失败不可评{sum(not p["reviewable"] for p in pairs)}对。')
    else:
        print('匿名包：待生成。')
    summary_path = folder / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary['ratings_sha256'] != digest(folder / 'review/ratings.csv'):
            print('报告：评分已修改，需要重新report。')
        else:
            print('报告：' + ('完整评测已保存。' if summary['complete'] else '已保存诊断报告，有效样本不足，不能标为评测完成。'))
    else:
        print('完整报告：待人工评分与report。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare', 'generate', 'blind', 'report', 'status'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reviewer', help='report：实际评分者昵称或作者本人')
    parser.add_argument('--allow-partial', action='store_true', help='report：只生成未完成预览，不标为完整评测')
    args = parser.parse_args()
    if args.resume and args.step != 'generate':
        parser.error('--resume只适用于generate')
    if (args.reviewer or args.allow_partial) and args.step != 'report':
        parser.error('--reviewer/--allow-partial只适用于report')
    try:
        config_path = args.config.resolve()
        if args.step == 'generate':
            generate(config_path, args.resume)
        elif args.step == 'report':
            report(config_path, args.reviewer, args.allow_partial)
        else:
            {'prepare': prepare, 'blind': blind, 'status': status}[args.step](config_path)
    except KeyboardInterrupt:
        raise SystemExit('已中断；已写入的回答保留，使用generate --resume继续。')
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit('M5步骤未通过：' + str(error))


if __name__ == '__main__':
    main()
