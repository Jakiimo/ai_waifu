"""正式路线数据隔离、SFT目标与真实临时Trainer检查；不训练Qwen。"""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from scripts import m6,m6_train,m6_dpo
from preference_data import encode_preference


class FormalTests(unittest.TestCase):
    def assets(self):
        root=m6.ROOT;freeze=m6.read_json(root/'data/m6/FORMAL_FREEZE.json')
        cases=m6.read_jsonl(root/'data/m6/scenarios.jsonl');prompt=m6.read_json(root/'configs/m2_prompts.json')['short']
        decisions=m6.read_jsonl(root/freeze['decision_snapshot']);raw=m6.read_jsonl(root/freeze['candidate_snapshot'])
        packs={p['pack_id']:p for p in (m6.read_json(root/n) for n in freeze['pack_files'])}
        train=m6.read_jsonl(root/freeze['split_files']['train']);dev=m6.read_jsonl(root/freeze['split_files']['dev'])
        return train,dev,cases,prompt,decisions,packs,raw

    def test_final_data_retains_review_binding_and_disjoint_splits(self):
        train,dev,cases,prompt,decisions,packs,raw=self.assets()
        m6_train.validate_rows(train,'train',cases,prompt,decisions,packs,raw)
        m6_train.validate_rows(dev,'dev',cases,prompt,decisions,packs,raw)
        self.assertEqual((len(train),len(dev)),(179,6))
        self.assertFalse({r['family_id'] for r in train}&{r['family_id'] for r in dev})
        bad=deepcopy(train);bad[0]['rejected']='换成不属于采样的回复'
        with self.assertRaises(ValueError):m6_train.validate_rows(bad,'train',cases,prompt,decisions,packs,raw)
        with self.assertRaises(ValueError):m6_train.validate_rows(dev,'train',cases,prompt,decisions,packs,raw)

    def test_development_revision_and_unknown_test_rows_are_rejected(self):
        train,dev,cases,prompt,decisions,packs,raw=self.assets()
        bad=deepcopy(dev);bad[0]['chosen_is_revision']=True
        with self.assertRaisesRegex(ValueError,'开发chosen'):m6_train.validate_rows(bad,'dev',cases,prompt,decisions,packs,raw)
        bad=deepcopy(train);bad[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'划分'):m6_train.validate_rows(bad,'train',cases,prompt,decisions,packs,raw)

    def test_chosen_sft_masks_all_context_and_excludes_rejected_with_one_eos(self):
        from transformers import AutoTokenizer
        train,*_=self.assets();t=AutoTokenizer.from_pretrained(m6.ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
        for row,encoded in zip(train,m6_train.sft_rows(t,train,320)):
            pair=encode_preference(t,row['prompt'],row['chosen'],row['rejected'],320)
            count=len(pair['prompt_input_ids'])
            self.assertTrue(all(v==-100 for v in encoded['labels'][:count]))
            self.assertEqual(encoded['labels'][count:],pair['chosen_input_ids'])
            self.assertEqual(encoded['input_ids'],pair['prompt_input_ids']+pair['chosen_input_ids'])
            self.assertEqual(encoded['labels'].count(t.eos_token_id),1)

    @unittest.skipUnless(importlib.util.find_spec('trl'),'需独立M6环境')
    def test_actual_both_trainers_handle_partial_accumulation_and_only_change_policy(self):
        import torch
        from peft import LoraConfig,get_peft_model,PeftModel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
        torch.manual_seed(42)
        token=Tokenizer(WordLevel({'[PAD]':0,'[EOS]':1,'[UNK]':2,'p':3,'good':4,'bad':5},unk_token='[UNK]'))
        token.pre_tokenizer=Whitespace()
        t=PreTrainedTokenizerFast(tokenizer_object=token,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
        t.chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}"
        base=GPT2LMHeadModel(GPT2Config(vocab_size=6,n_layer=1,n_head=1,n_embd=8,n_positions=32,bos_token_id=1,eos_token_id=1,pad_token_id=0))
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B' in name:p.fill_(0.02)
        rows=[{'id':f'temporary-qa-{i}','prompt':[{'role':'system','content':'p'},{'role':'user','content':'p'}],
               'chosen':'good','rejected':'bad'} for i in range(5)]
        cfg={**m6.read_json(m6_train.CONFIG),'max_length':32,'learning_rate':1e-3}
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);model.save_pretrained(root/'sft',save_embedding_layers=False);base=model.unload()
            plain_config=deepcopy(base.config)
            plain_state={n:p.detach().clone() for n,p in base.state_dict().items()}
            for method in ['chosen_sft','dpo']:
                with self.subTest(method=method):
                    base=GPT2LMHeadModel(deepcopy(plain_config));base.load_state_dict(plain_state)
                    model=PeftModel.from_pretrained(base,root/'sft',adapter_name='policy',is_trainable=True)
                    model.load_adapter(root/'sft',adapter_name='reference',is_trainable=False)
                    model.set_adapter('policy');m6_dpo.copy_reference_from_policy(model)
                    before={n:p.detach().clone() for n,p in model.named_parameters()}
                    trainer=m6_train.make_trainer(method,model,t,rows,rows[:1],cfg,root/method,force_cpu=True)
                    trainer.train()
                    self.assertEqual(trainer.state.global_step,2)  # 5样本/累积4，末组仍更新。
                    self.assertTrue(math.isfinite(trainer.evaluate()['eval_loss']))
                    changes=[]
                    for n,p in model.named_parameters():
                        if not torch.equal(before[n],p):changes.append(n);self.assertIn('.policy.',n)
                        if '.reference.' in n:self.assertFalse(p.requires_grad)
                    self.assertTrue(changes)
                    model.set_adapter('policy');model.eval()
                    ids=torch.tensor([[3,3]])
                    with torch.no_grad():expected=model(input_ids=ids).logits.detach().clone()
                    model.save_pretrained(root/(method+'_saved'),selected_adapters=['policy'],save_embedding_layers=False)
                    fresh=GPT2LMHeadModel(deepcopy(plain_config));fresh.load_state_dict(plain_state)
                    loaded=PeftModel.from_pretrained(fresh,root/(method+'_saved')/'policy',is_trainable=False);loaded.eval()
                    with torch.no_grad():actual=loaded(input_ids=ids).logits
                    self.assertTrue(torch.allclose(expected,actual,rtol=1e-5,atol=1e-6))


if __name__=='__main__':unittest.main()
