"""M5仅用假引擎和虚构QA答卷检查流程；不会生成正式测试集回答。"""
from contextlib import redirect_stdout
import asyncio
from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import scripts.m5 as m5


def cases_fixture(count=4):
    return [{'id': f'qa-{i}', 'category': 'technical' if i == 0 else 'daily',
             'messages': [{'role': 'user', 'content': '前文问题'},
                          {'role': 'assistant', 'content': '预置前文'},
                          {'role': 'user', 'content': f'新的QA问题{i}'}],
             'expectation': '检查重点不应送入模型'} for i in range(count)]


class FakeEngine:
    def __init__(self, interrupt_at=None, fail_at=None):
        self.calls = []
        self.lock = threading.RLock()
        self.interrupt_at, self.fail_at = interrupt_at, fail_at

    def ensure_loaded(self):
        pass

    def _generate(self, messages, mode):
        self.calls.append((deepcopy(messages), mode))
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt()
        if len(self.calls) == self.fail_at:
            raise RuntimeError('模拟失败')
        return {'mode': mode, 'answer': 'QA用回复。', 'input_tokens': 20,
                'output_tokens': 5, 'finish_reason': 'eos', 'seconds': 0.01}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = m5.read_json(m5.CONFIG)
        self.cfg['evaluation_id'] = 'm5_qa_only'
        self.cases = cases_fixture()
        self.rubric = m5.read_json(m5.ROOT / self.cfg['rubric_file'])
        self.tasks = m5.build_tasks(self.cases, {'short': '短设定', 'strong_examples': '长设定'}, self.cfg)
        self.folder = self.root / 'results/m5_qa_only'
        self.private = self.root / 'outputs/m5_qa_only'
        self.folder.mkdir(parents=True)
        self.private.mkdir(parents=True)
        self.manifest = {'conditions': {}, 'input_token_counts': [{'key': t['key'], 'input_tokens': 20} for t in self.tasks]}
        m5.dump_json(self.folder / 'manifest.json', self.manifest)
        (self.root / 'docs').mkdir()
        (self.root / 'docs/凛音_角色卡.md').write_text('QA角色设定')

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self, engine=None, resume=False):
        with redirect_stdout(io.StringIO()):
            return m5.generate_records(self.folder, self.cfg, self.tasks, engine or FakeEngine(), resume)

    def complete(self, engine=None):
        records = self.generate(engine)
        m5.dump_json(self.folder / 'generation_complete.json', {
            'generations_sha256': m5.digest(self.folder / 'generations.jsonl'),
            'manifest_sha256': m5.digest(self.folder / 'manifest.json')})
        return records

    def make_blind(self):
        with patch.object(m5, 'ROOT', self.root), \
             patch.object(m5, 'prepare', return_value=(self.cfg, self.cases, self.rubric, self.tasks, self.manifest)), \
             redirect_stdout(io.StringIO()):
            m5.blind()
        return m5.read_json(self.private / 'blind_key.json')['pairs']

    def fill_ratings(self, pairs):
        rows = []
        for pair in pairs:
            row = dict.fromkeys(m5.RATING_FIELDS, '')
            row.update(pair_id=pair['pair_id'], reviewable='yes' if pair['reviewable'] else 'no')
            if pair['reviewable']:
                for side in ['left', 'right']:
                    row[side + '_style'] = str({'A': 2, 'B': 3, 'C': 4, 'D': 5}[pair[side + '_group']])
                    row[side + '_usefulness'] = '4'
                    row[side + '_fact_error'] = 'no'
                    row[side + '_constraint_error'] = 'no'
                row['preference'] = 'left' if pair['left_group'] in ('C', 'D') else 'right'
            rows.append(row)
        m5.write_csv(self.folder / 'review/ratings.csv', rows)
        return rows

    def test_tasks_share_context_within_ac_bd_and_exclude_expectation(self):
        for i in range(0, len(self.tasks), 4):
            a, b, c, d = self.tasks[i:i+4]
            self.assertEqual(a['messages'], c['messages'])
            self.assertEqual(b['messages'], d['messages'])
            self.assertEqual(a['messages'][1:], self.cases[i//4]['messages'])
            self.assertNotIn('检查重点', str(a['messages']))
            self.assertNotIn('expectation', a)

    def test_interruption_resume_and_completed_cache_do_not_duplicate_answers(self):
        with self.assertRaises(KeyboardInterrupt):
            self.generate(FakeEngine(interrupt_at=3))
        self.assertEqual(len(m5.read_generations(self.folder, self.tasks)), 2)
        with self.assertRaisesRegex(ValueError, 'resume'):
            self.generate()
        engine = FakeEngine()
        records = self.generate(engine, resume=True)
        self.assertEqual(len(engine.calls), len(self.tasks) - 2)
        self.assertEqual(len(records), len(self.tasks))
        cached = FakeEngine()
        self.generate(cached)
        self.assertEqual(cached.calls, [])

    def test_errors_are_persisted_and_not_silently_retried(self):
        records = self.complete(FakeEngine(fail_at=2))
        self.assertEqual(sum(r['status'] == 'error' for r in records), 1)
        pairs = self.make_blind()
        self.assertEqual(sum(not p['reviewable'] for p in pairs), 1)
        self.fill_ratings(pairs)
        rated, pending = m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)
        self.assertEqual(pending, [])
        summary = m5.summarize(records, pairs, rated, self.cases, self.rubric,
                               {t['key']:20 for t in self.tasks})
        self.assertEqual(summary['automatic']['B']['errors'], 1)
        self.assertEqual(summary['automatic']['B']['input_token_denominator'], 4)
        self.assertEqual(summary['pairwise']['BD']['generation_failed_pairs'], 1)
        self.assertEqual(summary['pairwise']['BD']['rated_pairs'], 3)

    def test_duplicate_or_changed_context_records_are_rejected(self):
        records = self.generate()
        path = self.folder / 'generations.jsonl'
        with path.open('a') as handle:
            handle.write(m5.json.dumps(records[0]) + '\n')
        with self.assertRaisesRegex(ValueError, '重复'):
            m5.read_generations(self.folder, self.tasks)
        records[0]['messages'] = []
        path.write_text('\n'.join(m5.json.dumps(r) for r in records)+'\n')
        with self.assertRaisesRegex(ValueError, '条件'):
            m5.read_generations(self.folder, self.tasks)

    def test_balanced_randomization_is_reproducible_and_each_answer_used_once(self):
        records = self.generate()
        pairs = m5.build_pairs(records, self.cases, 123)
        self.assertEqual(pairs, m5.build_pairs(records, self.cases, 123))
        self.assertNotEqual(pairs, m5.build_pairs(records, self.cases, 124))
        keys = [p[side+'_key'] for p in pairs for side in ('left', 'right')]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {t['key'] for t in self.tasks})
        for name, adapted in [('AC','C'), ('BD','D')]:
            self.assertEqual(sum(p['comparison']==name and p['left_group']==adapted for p in pairs), 2)

    def test_blind_material_has_no_group_labels_and_rerun_preserves_edited_ratings(self):
        self.complete()
        pairs = self.make_blind()
        book = (self.folder / 'review/batch_01.md').read_text()
        for phrase in ['left_group', 'right_group', 'strong_examples', '原模型', 'LoRA', 'group']:
            self.assertNotIn(phrase, book)
        self.fill_ratings(pairs)
        before = (self.folder / 'review/ratings.csv').read_bytes()
        self.make_blind()
        self.assertEqual((self.folder / 'review/ratings.csv').read_bytes(), before)

    def test_blank_scores_stay_pending_and_invalid_scores_are_not_ignored(self):
        self.complete()
        pairs = self.make_blind()
        rated, pending = m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)
        self.assertEqual(rated, [])
        self.assertEqual(len(pending), len(pairs))
        rows = self.fill_ratings(pairs)
        rows[0]['left_style'] = '9'
        m5.write_csv(self.folder / 'review/ratings.csv', rows)
        with self.assertRaisesRegex(ValueError, '无效'):
            m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)

    def test_csv_duplicate_and_technical_na_are_rejected(self):
        self.complete()
        pairs = self.make_blind()
        rows = self.fill_ratings(pairs)
        rows[0]['pair_id'] = rows[1]['pair_id']
        m5.write_csv(self.folder / 'review/ratings.csv', rows)
        with self.assertRaisesRegex(ValueError, '重复'):
            m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)
        rows = self.fill_ratings(pairs)
        index = next(i for i,p in enumerate(pairs) if p['case_id']=='qa-0')
        rows[index]['left_fact_error'] = 'na'
        m5.write_csv(self.folder / 'review/ratings.csv', rows)
        with self.assertRaisesRegex(ValueError, '技术题'):
            m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)

    def test_orientation_aware_summary_and_win_rate_denominators(self):
        records = self.complete()
        pairs = self.make_blind()
        rows = self.fill_ratings(pairs)
        bd_indexes = [i for i,p in enumerate(pairs) if p['comparison']=='BD']
        rows[bd_indexes[0]]['preference'] = 'tie'
        p = pairs[bd_indexes[1]]
        rows[bd_indexes[1]]['preference'] = 'left' if p['left_group']=='B' else 'right'
        m5.write_csv(self.folder / 'review/ratings.csv', rows)
        rated, pending = m5.parse_ratings(self.folder / 'review/ratings.csv', pairs, self.cases)
        summary = m5.summarize(records, pairs, rated, self.cases, self.rubric)
        self.assertEqual(summary['pairwise']['BD']['lora_wins'], 2)
        self.assertEqual(summary['pairwise']['BD']['ties'], 1)
        self.assertEqual(summary['pairwise']['BD']['lora_losses'], 1)
        self.assertEqual(summary['pairwise']['BD']['win_rate_all_rated'], 0.5)
        self.assertEqual(summary['pairwise']['BD']['win_rate_excluding_ties'], 0.6667)
        self.assertEqual(summary['human']['D']['mean_style'], 5)
        self.assertEqual(summary['human']['B']['mean_style'], 3)
        self.assertEqual(summary['human']['D']['rated_answers'], 4)
        self.assertEqual(summary['secondary_C_vs_B']['mean_C_minus_B_style'], 1)

    def test_report_requires_real_scores_and_partial_report_is_explicit(self):
        self.complete()
        pairs = self.make_blind()
        with patch.object(m5,'ROOT',self.root), \
             patch.object(m5,'prepare',return_value=(self.cfg,self.cases,self.rubric,self.tasks,self.manifest)), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'未完整评分'):
                m5.report(reviewer='QA模拟评分者')
            m5.report(reviewer='QA模拟评分者',allow_partial=True)
            self.assertFalse(m5.read_json(self.folder/'summary_partial.json')['complete'])
            self.assertFalse((self.folder/'summary.json').exists())
            self.fill_ratings(pairs)
            m5.report(reviewer='QA模拟评分者')
        report = m5.read_json(self.folder/'summary.json')
        self.assertTrue(report['ratings_complete'])
        self.assertFalse(report['complete'])  # 只有4题QA，不能冒充至少40题的正式比较。
        self.assertEqual(report['reviewer_count'],1)
        self.assertEqual(report['pairwise']['BD']['rated_pairs'],4)

    def test_all_failed_generations_cannot_be_marked_complete(self):
        records=[]
        for task in self.tasks:
            records.append({**task,'status':'error','answer':'','finish_reason':'error',
                            'error_type':'QAError','error':'模拟全部失败','input_tokens':None,'output_tokens':0})
        (self.folder/'generations.jsonl').write_text(''.join(m5.json.dumps(r)+'\n' for r in records))
        m5.dump_json(self.folder/'generation_complete.json',{
            'generations_sha256':m5.digest(self.folder/'generations.jsonl'),
            'manifest_sha256':m5.digest(self.folder/'manifest.json')})
        self.make_blind()
        with patch.object(m5,'ROOT',self.root), \
             patch.object(m5,'prepare',return_value=(self.cfg,self.cases,self.rubric,self.tasks,self.manifest)), \
             redirect_stdout(io.StringIO()):
            m5.report(reviewer='QA模拟评分者')
        summary=m5.read_json(self.folder/'summary.json')
        self.assertTrue(summary['ratings_complete'])
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['quality_comparison_usable'])
        self.assertEqual(summary['human']['D']['rated_answers'],0)

    def test_modified_blind_key_or_raw_answers_fail_integrity(self):
        self.complete()
        self.make_blind()
        path=self.private/'blind_key.json'
        original=path.read_text()
        path.write_text(original+' ')
        with patch.object(m5,'ROOT',self.root), self.assertRaisesRegex(ValueError,'身份映射'):
            m5.check_blind_integrity(self.folder,self.private)
        path.write_text(original)
        with (self.folder/'generations.jsonl').open('a') as handle:handle.write('\n')
        with patch.object(m5,'ROOT',self.root), self.assertRaisesRegex(ValueError,'原始回答'):
            m5.check_blind_integrity(self.folder,self.private)

    def test_real_frozen_preflight_does_not_generate_or_change_test_data(self):
        test_path=m5.ROOT/self.cfg['test_file']
        before=m5.digest(test_path)
        cfg,cases,rubric,tasks,conditions=m5.preflight()
        self.assertEqual(len(cases),60)
        self.assertEqual(len(tasks),240)
        self.assertEqual(m5.digest(test_path),before)
        self.assertEqual(conditions['training_run_id'],'m3_learning_01')

    def test_review_save_is_atomic_and_invalid_rating_does_not_overwrite_csv(self):
        from scripts import m5_review
        self.complete()
        self.make_blind()
        items=m5.read_json(self.folder/'review/review_items.json')
        path=self.folder/'review/ratings.csv'
        item=next(item for item in items if item['category']=='technical')
        values=['4','4','3','3','no','no','no','no','left','QA备注']
        rated,pending=m5_review.save_rating(path,items,item['pair_id'],values)
        self.assertEqual(rated,1)
        self.assertEqual(len(pending),7)
        before=path.read_bytes()
        values[0]='9'
        with self.assertRaisesRegex(ValueError,'无效'):
            m5_review.save_rating(path,items,item['pair_id'],values)
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse((path.parent/'ratings.pending.csv').exists())

    def test_review_ui_has_no_default_scores_and_does_not_need_identity_key(self):
        from scripts import m5_review
        self.complete()
        self.make_blind()
        (self.private/'blind_key.json').unlink()  # UI视图不读取映射内容。
        async def construct():
            return m5_review.build_app(self.folder/'review')
        with patch.object(m5.demo,'ChatEngine',side_effect=AssertionError('评分网页不应加载模型')):
            app=asyncio.run(construct())
        radios=[block for block in app.blocks.values() if block.__class__.__name__=='Radio']
        self.assertEqual(len(radios),9)
        self.assertTrue(all(block.value is None for block in radios))
        public=m5.read_json(self.folder/'review/review_items.json')
        for item in public:
            self.assertNotIn('left_group',item)
            self.assertNotIn('right_group',item)
            self.assertNotIn('case_id',item)
        callbacks={fn.fn.__name__:fn.fn for fn in app.fns.values()}
        values=['4','4','4','4','no','no','no','no','tie','']
        callbacks['save_and_next']('pair-001',*values)
        pairs,cases=m5_review.public_pairs(public)
        rated,pending=m5.parse_ratings(self.folder/'review/ratings.csv',pairs,cases)
        self.assertEqual(len(rated),1)
        resumed=callbacks['resume_view']()
        dropdown=next(block for block in app.blocks.values() if block.__class__.__name__=='Dropdown')
        self.assertEqual(resumed[dropdown],'pair-002')


if __name__ == '__main__':
    unittest.main()
