"""M3接入检查：真实多轮监督、collator保留标签、开发集不进入训练及产物保护。"""
from copy import deepcopy
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.m3 as m3


class M3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.cfg = m3.read_json(m3.CONFIG)
        cls.tokenizer = AutoTokenizer.from_pretrained(m3.MODEL, local_files_only=True)
        cls.train, cls.dev, cls.counts = m3.datasets(cls.tokenizer, cls.cfg)

    def test_current_training_targets_and_history_are_encoded_without_dev_leakage(self):
        rows=m3.read_jsonl(m3.TRAIN)
        self.assertEqual(len(self.train),len(rows))
        self.assertEqual(len(self.dev),len(m3.read_jsonl(m3.DEV)))
        for row, batch, counts in zip(rows,self.train,self.counts):
            text=self.tokenizer.decode([x for x in batch['labels'] if x!=-100])
            self.assertEqual(text.split(self.tokenizer.eos_token)[0],row['messages'][-1]['content'])
            prefix=self.tokenizer.decode(batch['input_ids'][:counts['prompt_tokens']])
            for message in row['messages'][:-1]:self.assertIn(message['content'],prefix)
            self.assertTrue(all(x==-100 for x in batch['labels'][:counts['prompt_tokens']]))
        dev_ids={r['id'] for r in m3.read_jsonl(m3.DEV)}
        self.assertFalse(dev_ids & {r['id'] for r in rows})

    def test_collator_preserves_history_and_padding_ignore_labels(self):
        from transformers import default_data_collator
        import torch
        batch=default_data_collator([self.train[0],self.train[-1]])
        self.assertTrue(torch.equal(batch['labels'],torch.tensor([self.train[0]['labels'],self.train[-1]['labels']])))
        self.assertTrue(torch.all(batch['labels'][batch['attention_mask']==0]==-100))

    def test_dev_targets_are_frozen_reference_answers(self):
        cases=m3.read_jsonl(m3.DEV)
        for case,batch in zip(cases,self.dev):
            active=[x for x in batch['labels'] if x!=-100]
            self.assertEqual(self.tokenizer.decode(active).split(self.tokenizer.eos_token)[0],case['reference_answer'])

    def test_mismatched_model_and_dataset_are_rejected(self):
        m2=m3.read_json(m3.M2_CONFIG)
        for key,value in [('model_id','other/model'),('dataset_version','old-data')]:
            cfg=deepcopy(self.cfg);cfg[key]=value
            with self.assertRaisesRegex(ValueError,'与冻结M2不同'):m3.validate_config(cfg,m2)

    def test_run_id_cannot_escape_into_existing_outputs(self):
        for run_id in ['../m1_adapter','m1','m2','m1_adapter']:
            cfg=deepcopy(self.cfg);cfg['run_id']=run_id
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError,'run_id'):
                m3.paths(cfg)

    def test_dev_comparison_rejects_changed_device_dtype_or_runtime(self):
        packages={'torch':'same','transformers':'same'}
        record={'device':'mps','dtype':'torch.bfloat16','packages':packages}
        baseline={'device':'mps','weight_dtype':'torch.bfloat16','runtime':packages}
        with patch.object(m3,'versions',return_value=packages):
            m3.validate_comparison_environment(record,baseline,'mps','torch.bfloat16')
            with self.assertRaisesRegex(ValueError,'同一设备'):
                m3.validate_comparison_environment(record,baseline,'cpu','torch.bfloat16')
            with self.assertRaisesRegex(ValueError,'同一基础权重精度'):
                m3.validate_comparison_environment(record,baseline,'mps','torch.float32')
        with patch.object(m3,'versions',return_value={'torch':'changed','transformers':'same'}):
            with self.assertRaisesRegex(ValueError,'依赖'):
                m3.validate_comparison_environment(record,baseline,'mps','torch.bfloat16')


if __name__=='__main__':unittest.main()
