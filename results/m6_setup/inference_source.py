"""M6第一步：冻结场景与规则 → 采样SFT候选 → 匿名审核包 → 导出偏好对。

不自动训练DPO，不用M5测试题生成候选，不自动编造偏好标签。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from prepare_data import read_json, read_jsonl, dump_json, digest, normalize, similarity, PROMPTS
from preference_data import (build_candidate_tasks, validate_scenarios, object_digest,
                             validate_decision, encode_preference)
from scripts.m1 import now, versions
import demo

CONFIG = ROOT / 'configs/m6.json'
DATA = ROOT / 'data/m6'
RESULTS = ROOT / 'results/m6_preparation'
PRIVATE = ROOT / 'outputs/m6_preparation'


def m6_versions():
    values = versions()
    for name in ('trl', 'datasets'):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('请用.venv-m6/bin/python；该独立环境才安装了TRL与datasets。')
    if values['trl'] != '0.24.0' or values['datasets'] != '4.3.0':
        raise ValueError('请使用固定的M6依赖。')
    return values


def preflight():
    cfg = read_json(CONFIG)
    demo_cfg, train_cfg, adapter, prompts = demo.validate_assets(ROOT / cfg['source_demo_config'])
    if demo_cfg['training_config'] != cfg['source_training_config'] or cfg['prompt_name'] != 'short':
        raise ValueError('候选必须来自固定的m3_learning_01 SFT和short Prompt。')
    if cfg['candidates_per_case'] != 2 or cfg['sampling']['do_sample'] is not True:
        raise ValueError('本轮每题两条采样候选。')
    if cfg['sampling']['temperature'] <= 0 or not 0 < cfg['sampling']['top_p'] <= 1:
        raise ValueError('采样参数错误。')
    cases = read_jsonl(ROOT / cfg['scenario_file'])
    stats = validate_scenarios(cases, cfg)
    rubric = read_json(ROOT / cfg['rubric_file'])
    if rubric['version'] != cfg['rubric_version']:
        raise ValueError('评分规则版本不符。')
    files = [CONFIG, ROOT / cfg['scenario_file'], ROOT / cfg['rubric_file'], PROMPTS,
             ROOT / cfg['source_training_config'], adapter / 'adapter_model.safetensors',
             adapter / 'adapter_config.json', adapter.parent / 'run_manifest.json',
             ROOT / 'models/model_manifest.json', ROOT / 'requirements-m6.txt', ROOT / 'requirements-m6-lock.txt',
             Path(__file__), ROOT / 'preference_data.py', ROOT / 'scripts/create_m6_scenarios.py',
             ROOT / 'demo.py', ROOT / 'scripts/m3.py', ROOT / 'training_data.py',
             ROOT / 'configs/m4.json', ROOT / 'docs/凛音_角色卡.md']
    conditions = {'data_version': cfg['data_version'], 'rubric_version': cfg['rubric_version'],
                  'source_run': train_cfg['run_id'], 'packages': m6_versions(),
                  'source_device': read_json(adapter.parent / 'run_manifest.json')['device'],
                  'source_dtype': read_json(adapter.parent / 'run_manifest.json')['dtype'],
                  'files': {str(p.relative_to(ROOT)): digest(p) for p in files}}
    return cfg, cases, rubric, train_cfg, adapter, prompts['short'], stats, conditions


def check_duplicates(cases):
    old = read_jsonl(ROOT / 'data/m2/train.jsonl') + read_jsonl(ROOT / 'data/m2/dev_scenarios.jsonl') + \
          read_jsonl(ROOT / 'data/test_scenarios.jsonl')
    # 比较最后问题与完整前文；字符检查只是筛查，不证明全量语义独立。
    checked = []
    for index, case in enumerate(cases):
        previous = old + [c for c in cases[:index] if c['split'] != case['split']]
        for other in previous:
            q, p = case['messages'][-1]['content'], other['messages'][-1]['content']
            score = similarity(q, p)
            if normalize(q) == normalize(p) or score >= read_json(CONFIG)['near_duplicate_threshold']:
                checked.append({'new': case['id'], 'other': other['id'], 'similarity': round(score, 4)})
    return checked


def prepare():
    cfg, cases, rubric, train_cfg, adapter, prompt, stats, conditions = preflight()
    path = RESULTS / 'manifest.json'
    if path.exists():
        manifest = read_json(path)
        if manifest['conditions'] != conditions:
            raise ValueError('M6准备条件已变化，保留现有记录；不可覆盖或静默并入其他候选。')
        for relative, expected in manifest['snapshot_files'].items():
            if digest(PRIVATE / 'source_snapshot' / relative) != expected:
                raise ValueError('M6源码/数据快照已变化。')
        return cfg, cases, rubric, train_cfg, adapter, prompt, manifest
    if (RESULTS / 'candidates.jsonl').exists():
        raise ValueError('已有候选但无manifest，不认定其来源。')
    hits = check_duplicates(cases)
    if hits:
        raise ValueError('存在旧数据或跨集合近重复，需先排查：' + str(hits[:5]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen2.5-1.5b-instruct', local_files_only=True)
    token_counts = []
    for case in cases:
        messages = [{'role': 'system', 'content': prompt}] + case['messages']
        count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if count > cfg['max_prompt_tokens']:
            raise ValueError('场景输入超长，不能静默截断：' + case['id'])
        token_counts.append({'id': case['id'], 'input_tokens': count})
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    snapshot = {}
    for relative, expected in conditions['files'].items():
        if relative.startswith(('models/', 'outputs/')):
            continue
        target = PRIVATE / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        snapshot[relative] = expected
    manifest = {'prepared_at_utc': now(), 'conditions': conditions, 'scenario_stats': stats,
                'input_token_counts': token_counts, 'max_input_tokens': max(r['input_tokens'] for r in token_counts),
                'snapshot_files': snapshot, 'near_duplicate_hits': hits,
                'data_source': 'AI辅助原创场景，尚无模型候选或审核标签',
                'test_usage': '新最终测试封存，不采样训练候选；未证明全量语义独立',
                'formal_training': 'not_run', 'grading_source': 'AI，未冒充人类偏好'}
    dump_json(path, manifest)
    return cfg, cases, rubric, train_cfg, adapter, prompt, manifest


@contextmanager
def run_lock():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'RUNNING.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个M6采样进程正在运行。')
        yield


def all_tasks(cases, cfg, prompt):
    return build_candidate_tasks(cases, cfg, prompt, 'train', cfg['planned_counts']['train']) + \
           build_candidate_tasks(cases, cfg, prompt, 'dev', cfg['planned_counts']['dev'])


def read_candidates(path, tasks, conditions):
    if not path.exists():
        return []
    known = {task['key']: task for task in tasks}
    rows, seen = [], set()
    for line, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f'候选记录第{line}行损坏，保留文件，不悄悄跳过。')
        key = row.get('key')
        if key not in known or key in seen:
            raise ValueError('候选含未知或重复key：' + str(key))
        if row.get('source_conditions_sha256') != object_digest(conditions):
            raise ValueError('候选来源条件不符。')
        if any(row.get(k) != value for k, value in known[key].items()):
            raise ValueError('候选输入或seed与计划不同：' + key)
        if row.get('status') not in ('ok', 'error'):
            raise ValueError('候选生成状态错误。')
        if row['status'] == 'ok' and (not row.get('answer', '').strip() or row.get('finish_reason') not in ('eos', 'length')):
            raise ValueError('成功候选缺内容或结束原因。')
        rows.append(row)
        seen.add(key)
    return rows


class CandidateEngine:
    def __init__(self, cfg, train_cfg, adapter):
        self.cfg, self.train_cfg, self.adapter = cfg, train_cfg, adapter
        self.model = None

    def ensure_loaded(self):
        if self.model is None:
            from peft import PeftModel
            from scripts.m3 import load_base
            base, self.tokenizer, self.device = load_base(self.train_cfg)
            self.model = PeftModel.from_pretrained(base, self.adapter, is_trainable=False, local_files_only=True)
            self.model.requires_grad_(False)
            self.model.eval()
            self.model.config.use_cache = True
            source = read_json(self.adapter.parent / 'run_manifest.json')
            if self.device != source['device'] or str(next(self.model.parameters()).dtype) != source['dtype']:
                raise ValueError('候选设备/精度与SFT起点不同；尚未生成或保存候选。')

    def generate(self, task):
        import torch
        from transformers import set_seed
        self.ensure_loaded()
        set_seed(task['seed'])
        ids = self.tokenizer.apply_chat_template(task['messages'], tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to(self.device)
        if ids.shape[1] > self.cfg['max_prompt_tokens']:
            raise ValueError('输入超长。')
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                **self.cfg['sampling'], use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        tokens = output[0, ids.shape[1]:]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not answer or not any(c.isalnum() for c in answer):
            raise RuntimeError('空或纯符号候选。')
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('候选生成应全部冻结。')
        return {'answer': answer, 'input_tokens': ids.shape[1], 'output_tokens': tokens.numel(),
                'finish_reason': 'eos' if int(tokens[-1]) == self.tokenizer.eos_token_id else 'length',
                'seconds': round(time.perf_counter() - started, 3)}


def generate_candidates(path, tasks, known_tasks, conditions, engine, resume=False):
    rows = read_candidates(path, known_tasks, conditions)
    seen = {r['key'] for r in rows}
    selected = {t['key'] for t in tasks}
    if selected <= seen:
        print('本批候选已齐备，未重复生成。', flush=True)
        return rows
    if selected & seen and not resume:
        raise ValueError('本批已有部分候选，请加--resume；不覆盖或重复生成。')
    engine.ensure_loaded()
    for index, task in enumerate(tasks, 1):
        if task['key'] in seen:
            continue
        try:
            result = engine.generate(task)
            row = {**task, **result, 'status': 'ok'}
        except KeyboardInterrupt:
            raise
        except Exception as error:
            row = {**task, 'status': 'error', 'answer': '', 'finish_reason': 'error',
                   'error_type': type(error).__name__, 'error': str(error)}
        row.update(source_conditions_sha256=object_digest(conditions), recorded_at_utc=now())
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        rows.append(row)
        print(f'{index}/{len(tasks)} {task["key"]}：{row["status"]}，{row["finish_reason"]}', flush=True)
    return rows


def collect(split, limit, resume=False):
    cfg, cases, rubric, train_cfg, adapter, prompt, manifest = prepare()
    tasks = build_candidate_tasks(cases, cfg, prompt, split, limit)
    with run_lock():
        engine = CandidateEngine(cfg, train_cfg, adapter)
        rows = generate_candidates(RESULTS / 'candidates.jsonl', tasks, all_tasks(cases, cfg, prompt),
                                   manifest['conditions'], engine, resume)
        if engine.model is not None:
            if engine.device != manifest['conditions']['source_device'] or \
               str(next(engine.model.parameters()).dtype) != manifest['conditions']['source_dtype']:
                raise ValueError('候选设备/精度与SFT来源不同，不并入正式偏好数据。')
        if preflight()[-1] != manifest['conditions']:
            raise ValueError('采样期间输入来源发生变化。')
        dump_json(RESULTS / 'collection_progress.json', {'recorded_at_utc': now(), 'saved': len(rows),
            'errors': sum(r['status'] == 'error' for r in rows), 'split': split, 'requested_cases': limit,
            'training_performed': False, 'adapter_unchanged': True})
    print('候选已保存；不是训练，也没有自动标成chosen/rejected。下一步review-pack。')


def build_review_items(rows, cases, seed):
    by_key = {row['key']: row for row in rows}
    rng = random.Random(seed)
    items = []
    for case in cases:
        if case['split'] == 'test':
            continue
        keys = [case['id'] + f':candidate-{i}' for i in range(2)]
        if not all(key in by_key for key in keys):
            continue
        pair = [by_key[key] for key in keys]
        digest_value = object_digest(pair)
        rng.shuffle(pair)
        ok = all(row['status'] == 'ok' for row in pair)
        different = pair[0]['answer'].strip() != pair[1]['answer'].strip()
        item = {'case_id': case['id'], 'split': case['split'], 'category': case['category'],
                'messages': case['messages'], 'expectation': case['expectation'],
                'candidates_sha256': digest_value, 'reviewable': ok and different,
                'issue': '' if ok and different else '生成失败' if not ok else '两侧完全相同',
                'left_answer': pair[0]['answer'] or '〔生成失败〕',
                'right_answer': pair[1]['answer'] or '〔生成失败〕',
                'left_finish_reason': pair[0]['finish_reason'], 'right_finish_reason': pair[1]['finish_reason']}
        items.append((item, {'case_id': case['id'], 'left_key': pair[0]['key'], 'right_key': pair[1]['key']}))
    return items


def review_pack(split):
    cfg, cases, rubric, train_cfg, adapter, prompt, manifest = prepare()
    rows = read_candidates(RESULTS / 'candidates.jsonl', all_tasks(cases, cfg, prompt), manifest['conditions'])
    selected = [case for case in cases if case['split'] == split]
    items_keys = build_review_items(rows, selected, cfg['seed'])
    if not items_keys:
        raise ValueError('没有完整两条候选的场景，先collect。')
    items, keys = map(list, zip(*items_keys))
    pack_id = split + '_' + object_digest(items)[:16]
    path = RESULTS / 'review' / (pack_id + '.json')
    key_path = PRIVATE / 'review_keys' / (pack_id + '.json')
    package = {'pack_id': pack_id, 'rubric_version': cfg['rubric_version'],
               'source_conditions_sha256': object_digest(manifest['conditions']), 'items': items}
    if path.exists():
        if read_json(path) != package or read_json(key_path) != keys:
            raise ValueError('匿名审核包或对应映射已变化。')
    else:
        dump_json(path, package)
        dump_json(key_path, keys)
    lines = ['# M6 偏好审核包', '', '这是SFT候选，不是DPO模型输出；身份隐藏，左右随机。评分交给助手，学习者无需逐题标注。', '',
             '评分规则：configs/m6_rubric.json；先事实、要求与帮助，再人设和口吻。双方差就拒收或修订。', '']
    for item in items:
        lines += ['## ' + item['case_id'], '', '类别：' + item['category'], '']
        for message in item['messages']:
            lines += [message['role'] + '：' + message['content'], '']
        lines += ['检查重点：' + item['expectation'], '', '### 左', '', item['left_answer'], '',
                  '### 右', '', item['right_answer'], '', '生成问题：' + (item['issue'] or '无生成错误；质量待审核'), '']
    md_path = path.with_suffix('.md')
    text = '\n'.join(lines)
    if md_path.exists() and md_path.read_text(encoding='utf-8') != text:
        raise ValueError('已有审核阅读版变化，不覆盖。')
    md_path.write_text(text, encoding='utf-8')
    print(f'审核包：{path.relative_to(ROOT)}，{len(items)}个场景；交给助手审核，不需要你打分。')


def export_pairs():
    cfg, cases, rubric, train_cfg, adapter, prompt, manifest = prepare()
    decision_file = DATA / 'review_decisions.jsonl'
    if not decision_file.exists():
        raise ValueError('还没有助手实际审核记录；不能凭空导出偏好对。')
    rows = read_candidates(RESULTS / 'candidates.jsonl', all_tasks(cases, cfg, prompt), manifest['conditions'])
    raw_by_key = {row['key']: row for row in rows}
    case_by_id = {case['id']: case for case in cases}
    seen, exported, rejected = set(), [], []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen2.5-1.5b-instruct', local_files_only=True)
    for decision in read_jsonl(decision_file):
        if decision['case_id'] in seen:
            raise ValueError('同一case重复审核，不静默取最后一条。')
        seen.add(decision['case_id'])
        pack_id = decision['pack_id']
        if not isinstance(pack_id, str) or not all(c.isalnum() or c=='_' for c in pack_id):
            raise ValueError('pack_id格式错误。')
        package = read_json(RESULTS / 'review' / (pack_id + '.json'))
        if package['source_conditions_sha256'] != object_digest(manifest['conditions']):
            raise ValueError('审核包属于其他实验条件。')
        item = next(i for i in package['items'] if i['case_id'] == decision['case_id'])
        case = case_by_id[item['case_id']]
        if case['split'] == 'test' or item['messages'] != case['messages'] or item['expectation'] != case['expectation']:
            raise ValueError('审核前文或划分与计划不同。')
        keys = next(k for k in read_json(PRIVATE / 'review_keys' / (pack_id + '.json')) if k['case_id'] == item['case_id'])
        raw = [raw_by_key[case['id'] + f':candidate-{i}'] for i in range(2)]
        if object_digest(raw) != item['candidates_sha256'] or any(
            raw_by_key[keys[s + '_key']]['answer'] != item[s + '_answer'] for s in ('left', 'right') if item['reviewable']):
            raise ValueError('审核包与实际候选不一致。')
        accepted = validate_decision(decision, item)
        if accepted is None:
            rejected.append({'case_id': case['id'], 'reason': decision['reason']})
            continue
        messages = [{'role': 'system', 'content': prompt}] + case['messages']
        encode_preference(tokenizer, messages, accepted['chosen'], accepted['rejected'], cfg['max_pair_sequence_tokens'])
        exported.append({'id': case['id'], 'family_id': case['family_id'], 'split': case['split'],
            'category': case['category'], 'data_version': cfg['data_version'], 'rubric_version': cfg['rubric_version'],
            'prompt': messages, **accepted, 'preference_reason': decision['reason'],
            'failure_tags': decision.get('failure_tags', []), 'review': decision,
            'candidate_source': {'conditions': manifest['conditions'], 'candidates_sha256': item['candidates_sha256'],
                                 'chosen_key': keys[accepted['chosen_side'] + '_key'],
                                 'revision_source': decision.get('revision_source')},
            'review_decision_sha256': object_digest(decision)})
    # 原始候选和审核记录不改；每次导出保留独立快照，不覆盖此前版本。
    version = object_digest({'pairs': exported, 'decisions_sha256': digest(decision_file)})[:16]
    target = DATA / 'pairs' / version
    target.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev'):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in exported if row['split'] == split)
        path = target / (split + '.jsonl')
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError('已有偏好快照被修改。')
        path.write_text(content, encoding='utf-8')
    dump_json(target / 'manifest.json', {'created_at_utc': now(), 'accepted_counts': dict(Counter(r['split'] for r in exported)),
        'rejected': rejected, 'reviewed_cases': len(seen), 'planned_counts': cfg['planned_counts'],
        'all_planned_train_dev_reviewed': len(seen) == cfg['planned_counts']['train'] + cfg['planned_counts']['dev'],
        'formal_training_ready': False, 'note': '审核后的偏好快照；仍需训练前数据/预算冻结与资源验证',
        'conditions': manifest['conditions'], 'decisions_sha256': digest(decision_file)})
    print('已导出真实审核偏好快照：' + str(target.relative_to(ROOT)) + '；不代表已训练。')


def status():
    cfg, cases, rubric, train_cfg, adapter, prompt, manifest = prepare()
    rows = read_candidates(RESULTS / 'candidates.jsonl', all_tasks(cases, cfg, prompt), manifest['conditions'])
    print('M6场景计划：训练180，开发30，最终测试60；最终测试不采样训练候选。')
    print('SFT来源：' + train_cfg['run_id'] + '；最大输入：' + str(manifest['max_input_tokens']) + ' token。')
    for split in ('train', 'dev'):
        saved = [row for row in rows if row['split'] == split]
        print(f'{split}候选：{len(saved)}/{cfg["planned_counts"][split] * 2}；失败{sum(r["status"]=="error" for r in saved)}。')
    decision_path = DATA / 'review_decisions.jsonl'
    print('审核记录：' + str(len(read_jsonl(decision_path)) if decision_path.exists() else 0) + '；无审核不自动填chosen/rejected。')
    print('正式DPO/chosen-only SFT：尚未执行。当前教学步骤：收集小批候选，再交由助手审核。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare', 'collect', 'review-pack', 'export-pairs', 'status'])
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--limit', type=int, default=20, help='collect：本split取前多少个场景，不是新增数量')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.step != 'collect':
        parser.error('--resume只用于collect')
    try:
        if args.step == 'collect':
            collect(args.split, args.limit, args.resume)
        elif args.step == 'review-pack':
            review_pack(args.split)
        elif args.step == 'export-pairs':
            export_pairs()
        elif args.step == 'status':
            status()
        else:
            prepared = prepare()
            print('M6准备通过：270个场景，0条新候选；未训练。' if not (RESULTS/'candidates.jsonl').exists()
                  else 'M6准备条件核对通过，已有候选保留；未启动训练。')
    except KeyboardInterrupt:
        raise SystemExit('已中断；已保存候选保留，原命令加--resume继续。')
    except (ValueError, RuntimeError, OSError, KeyError, StopIteration) as error:
        raise SystemExit('M6未通过：' + str(error))


if __name__ == '__main__':
    main()
