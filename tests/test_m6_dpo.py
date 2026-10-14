"""第二步审核绑定与训练框架测试；不训练真实Qwen或生成最终测试回答。"""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts import m6,m6_dpo


class DPOTests(unittest.TestCase):
    def test_reviewed_pairs_reject_answer_split_family_or_duplicate_tampering(self):
        cfg=m6.read_json(m6_dpo.CONFIG)
        rows=m6.read_jsonl(m6.ROOT/cfg['pair_file'])
        cases=m6.read_jsonl(m6.ROOT/'data/m6/scenarios.jsonl')
        prompt=m6.read_json(m6.ROOT/'configs/m2_prompts.json')['short']
        decisions=m6.read_jsonl(m6.ROOT/cfg['review_snapshot'])
        pack=m6.read_json(m6.ROOT/cfg['review_pack'])
        raw=m6.read_jsonl(m6.ROOT/cfg['candidate_snapshot'])
        m6_dpo.validate_pairs(rows,cases,prompt,decisions,pack,raw)
        for field,value in [('chosen','篡改的回答'),('split','test'),('family_id','错的族')]:
            changed=deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError):m6_dpo.validate_pairs(changed,cases,prompt,decisions,pack,raw)
        with self.assertRaises(ValueError):m6_dpo.validate_pairs(rows+[rows[0]],cases,prompt,decisions,pack,raw)

    def test_exact_reference_copy_fixes_low_precision_rounding_before_training(self):
        import torch
        from torch import nn
        model=nn.Module();model.lora_A=nn.ModuleDict({'policy':nn.Linear(2,2,bias=False),'reference':nn.Linear(2,2,bias=False)})
        p=model.lora_A['policy'].weight;r=model.lora_A['reference'].weight
        with torch.no_grad():
            p.fill_(0.01234567);r.copy_(p.bfloat16().float())
        self.assertFalse(torch.equal(p,r))
        m6_dpo.copy_reference_from_policy(model)
        self.assertTrue(torch.equal(p,r));self.assertFalse(r.requires_grad)
        with torch.no_grad():p.add_(0.01)
        self.assertFalse(torch.equal(p,r))  # 不是共享同一份可变参数或每步同步。

    @unittest.skipUnless(importlib.util.find_spec('trl'),'需M6独立环境')
    def test_reference_context_restores_policy_even_if_reference_forward_raises(self):
        cls=m6_dpo.trainer_class();instance=cls.__new__(cls)
        class FakeModel:
            active='policy'
            def set_adapter(self,name):self.active=name
        model=FakeModel()
        instance.model=model;instance.model_adapter_name='policy';instance.ref_adapter_name='reference'
        instance.accelerator=SimpleNamespace(unwrap_model=lambda x:x)
        with self.assertRaisesRegex(RuntimeError,'temporary QA'):
            with instance.null_ref_context():
                self.assertEqual(model.active,'reference')
                raise RuntimeError('temporary QA')
        self.assertEqual(model.active,'policy')

    @unittest.skipUnless(importlib.util.find_spec('trl'),'需M6独立环境')
    def test_actual_trainer_uses_fp32_loss_and_keeps_reference_and_base_unchanged(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig,get_peft_model,PeftModel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
        from trl import DPOConfig
        torch.manual_seed(43)
        vocab={'[PAD]':0,'[EOS]':1,'[UNK]':2,'prompt':3,'good':4,'bad':5}
        token=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));token.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=token,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
        base=GPT2LMHeadModel(GPT2Config(vocab_size=6,n_layer=1,n_head=1,n_embd=8,n_positions=32,
            bos_token_id=1,eos_token_id=1,pad_token_id=0))
        sft=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
        with torch.no_grad():
            for n,p in sft.named_parameters():
                if 'lora_B' in n:p.fill_(0.02)
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);sft.save_pretrained(root/'sft',save_embedding_layers=False);plain=sft.unload()
            model=PeftModel.from_pretrained(plain,root/'sft',adapter_name='policy',is_trainable=True)
            model.load_adapter(root/'sft',adapter_name='reference',is_trainable=False)
            model.set_adapter('policy');m6_dpo.copy_reference_from_policy(model)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            args=DPOConfig(output_dir=str(root/'qa'),use_cpu=True,report_to='none',max_steps=1,
                learning_rate=1e-3,gradient_checkpointing=False,optim='adamw_torch',per_device_train_batch_size=1,
                model_adapter_name='policy',ref_adapter_name='reference',precompute_ref_log_probs=True,
                save_strategy='no',disable_tqdm=True,dataloader_pin_memory=False,max_prompt_length=None,
                max_completion_length=None,max_length=32,bf16=False,fp16=False)
            data=Dataset.from_dict({'prompt':['prompt ']*2,'chosen':['good']*2,'rejected':['bad']*2})
            trainer=m6_dpo.trainer_class()(model=model,args=args,processing_class=tokenizer,train_dataset=data)
            x=torch.tensor([-5.],dtype=torch.bfloat16);y=torch.tensor([-6.],dtype=torch.bfloat16)
            loss,_,_=trainer.dpo_loss(x,y,x,y)
            self.assertEqual(loss.dtype,torch.float32)
            self.assertAlmostEqual(float(loss),0.693147,places=5)
            trainer.train();changes=[]
            for n,p in model.named_parameters():
                if not torch.equal(before[n],p):changes.append(n);self.assertIn('.policy.',n);self.assertIn('lora_',n)
                if '.reference.' in n:self.assertFalse(p.requires_grad)
            self.assertTrue(changes)


if __name__=='__main__':unittest.main()
