"""回归检查：多轮上下文丢失、过期开发结果复用、主路径版本混用。"""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare_data as p
from training_data import encode_conversation
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m2_pipeline',ROOT/'scripts/m2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PipelineTests(unittest.TestCase):
    def test_expansion_retains_history_and_only_last_reply_is_supervised(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(p.MODEL,local_files_only=True)
        card=next(c for c in p.read_jsonl(p.RAW) if c.get('history'))
        rows,_=p.expand_cards([card])
        self.assertEqual(rows[0]['messages'][:-2],card['history'])
        encoded,counts=encode_conversation(tokenizer,p.read_json(p.PROMPTS)['short'],rows[0]['messages'],256)
        prompt_text=tokenizer.decode(encoded['input_ids'][:counts['prompt_tokens']])
        for message in card['history']:self.assertIn(message['content'],prompt_text)
        active=[token for token in encoded['labels'] if token!=-100]
        self.assertEqual(tokenizer.decode(active).split(tokenizer.eos_token)[0],card['assistant'])
        self.assertTrue(all(label==-100 for label in encoded['labels'][:counts['prompt_tokens']]))

    def test_equal_last_question_with_different_history_is_not_deduplicated(self):
        cards=[{'family_id':str(i),'category':'relationship','source':'test','review':'test',
                'history':[{'role':'user','content':question},{'role':'assistant','content':'嗯。'}],
                'user_variants':['你怎么看？'],'assistant':answer} for i,question,answer in
               [(1,'我完成了任务','不错'),(2,'我忘了交作业','先处理补交')]]
        rows,removed=p.expand_cards(cards)
        self.assertEqual(len(rows),2);self.assertEqual(removed,[])

    def test_identical_full_context_across_splits_is_rejected(self):
        history=[{'role':'user','content':'我给猫起名薄荷'},{'role':'assistant','content':'记下了'},
                 {'role':'user','content':'叫什么？'}]
        train={'id':'train','family_id':'train','messages':history+[{'role':'assistant','content':'薄荷'}]}
        dev={'id':'dev','family_id':'dev','messages':history}
        with self.assertRaisesRegex(ValueError,'跨集合输入重复'):
            p.check_isolation([train],[dev],[],.65)

    def test_old_probe_cannot_be_reused_with_same_paths(self):
        with self.assertRaisesRegex(ValueError,'版本过期'):
            m.validate_probe({'dataset_version':'m2-persona-v1','rubric_version':'persona-v1'})

    def test_missing_and_duplicate_probe_answers_are_rejected(self):
        cfg=p.read_json(p.CONFIG)
        report={'dataset_version':cfg['dataset_version'],'rubric_version':cfg['rubric_version'],
                'input_files':{str(path.relative_to(ROOT)):p.digest(path) for path in p.frozen_paths()},
                'model':p.read_json(ROOT/'models/model_manifest.json'),'records':[]}
        with self.assertRaisesRegex(ValueError,'缺少候选'):
            m.validate_probe(report)
        report['records']=[{'prompt_name':name,'case_id':cid} for name in p.read_json(p.PROMPTS) for cid in cfg['probe_ids']]
        m.validate_probe(report)
        report['records'].append(report['records'][0])
        with self.assertRaisesRegex(ValueError,'重复记录'):
            m.validate_probe(report)

    def test_provisional_prompt_cannot_be_frozen(self):
        import tempfile
        cfg=p.read_json(p.CONFIG)
        decision={'dataset_version':cfg['dataset_version'],'ready_to_freeze':False}
        def fake_read(path):
            return decision if Path(path).name=='prompt_decision.json' else cfg
        with tempfile.TemporaryDirectory() as folder, patch.object(m,'FROZEN',Path(folder)/'frozen.json'), \
             patch.object(m,'prepare'), patch.object(m,'validate_probe',side_effect=lambda report:report), \
             patch.object(m,'read_json',side_effect=fake_read):
            with self.assertRaisesRegex(ValueError,'暂定'):
                m.freeze()
            self.assertFalse(m.FROZEN.exists())

    def test_active_train_and_holdouts_use_same_role_contract(self):
        cfg=p.read_json(p.CONFIG);prompts=p.read_json(p.PROMPTS)
        self.assertEqual(cfg['dataset_version'],'m2-tsundere-v2')
        self.assertEqual(p.read_json(ROOT/'configs/m2_rubric.json')['version'],cfg['rubric_version'])
        self.assertTrue(all('凛音' in prompt for prompt in prompts.values()))
        train=p.read_jsonl(p.TRAIN);dev=p.read_jsonl(p.DEV);tests=p.read_jsonl(p.TEST)
        self.assertEqual((len(train),len(dev),len(tests)),(360,24,60))
        self.assertEqual(sum(len(row['messages'])>2 for row in train),60)
        for row in dev+tests:
            self.assertEqual(row['dataset_version'],cfg['dataset_version'])
            self.assertEqual(row['rubric_version'],cfg['rubric_version'])
        self.assertEqual(p.expand_cards(p.read_jsonl(p.RAW))[0],train)

if __name__=='__main__':unittest.main()
