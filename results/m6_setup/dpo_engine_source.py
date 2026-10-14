"""M6第二步：实际偏好检查 → 学习者启动20对DPO验证 → 独立重载。

inspect不加载模型；check-engine做前向及一次反向检查、不更新参数；train才更新policy LoRA。
原M6候选源代码保持冻结，新增训练入口独立记录来源。
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import m6
from preference_data import encode_preference, object_digest, validate_decision

CONFIG = ROOT / 'configs/m6_dpo_smoke.json'


def paths(cfg):
    name = cfg['run_id']
    if not name.startswith('m6_dpo_smoke_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError('run_id必须使用独立m6_dpo_smoke_名称。')
    return ROOT/'outputs'/name, ROOT/'results'/name


def validate_pairs(rows, cases, prompt, decisions, package, raw_candidates):
    by_case = {c['id']: c for c in cases}
    by_review = {r['case_id']: r for r in decisions}
    items = {i['case_id']: i for i in package['items']}
    seen = set()
    for row in rows:
        name = row['id']
        if name in seen or name not in by_case or name not in by_review or name not in items:
            raise ValueError('偏好id重复或缺少审核来源。')
        seen.add(name)
        case, review, item = by_case[name], by_review[name], items[name]
        if row['split'] != 'train' or case['split'] != 'train' or row['review'] != review:
            raise ValueError('偏好不是训练区或审核已改变。')
        if row['family_id'] != case['family_id'] or row['category'] != case['category'] or \
           row['prompt'] != [{'role':'system','content':prompt}]+case['messages']:
            raise ValueError('偏好输入与固定场景不同。')
        pair_raw = [r for r in raw_candidates if r['case_id'] == name]
        pair_raw.sort(key=lambda r: r['variant'])
        if len(pair_raw) != 2 or object_digest(pair_raw) != item['candidates_sha256']:
            raise ValueError('原始候选版本不同。')
        accepted = validate_decision(review, item)
        if accepted is None or any(row[k] != accepted[k] for k in accepted):
            raise ValueError('chosen/rejected与实际审核不一致。')
        if object_digest(review) != row['review_decision_sha256']:
            raise ValueError('审核记录哈希不同。')


def preflight(config=CONFIG):
    cfg = m6.read_json(config)
    paths(cfg)
    if cfg['training_examples'] != 20 or cfg['max_steps'] != 5 or cfg['batch_size'] != 1 or \
       cfg['gradient_accumulation_steps'] != 4 or cfg['beta'] <= 0 or cfg['learning_rate'] <= 0:
        raise ValueError('本入口固定20对、5次更新的学习验证，不冒充正式180题训练。')
    data_cfg, cases, rubric, source_cfg, source_adapter, prompt, prep = m6.prepare()
    rows = m6.read_jsonl(ROOT/cfg['pair_file'])
    if len(rows) != 20 or cfg['max_length'] != data_cfg['max_pair_sequence_tokens']:
        raise ValueError('偏好数量或长度上限不同。')
    receipt = m6.read_json(ROOT/cfg['review_receipt'])
    for key, filename in [('decisions_sha256','review_snapshot'),('original_candidates_sha256','candidate_snapshot'),('pack_sha256','review_pack')]:
        if m6.digest(ROOT/cfg[filename]) != receipt[key]:
            raise ValueError('审核/候选原始快照发生变化：'+filename)
    decisions = m6.read_jsonl(ROOT/cfg['review_snapshot'])
    package = m6.read_json(ROOT/cfg['review_pack'])
    raw = m6.read_candidates(ROOT/cfg['candidate_snapshot'],m6.all_tasks(cases,data_cfg,prompt),prep['conditions'])
    validate_pairs(rows,cases,prompt,decisions,package,raw)
    if m6.read_json(ROOT/cfg['pair_manifest'])['conditions'] != prep['conditions']:
        raise ValueError('偏好快照的SFT来源不同。')
    dev = {case['id']:case for case in cases if case['split']=='dev'}
    if len(cfg['development_case_ids']) != 6 or len(set(cfg['development_case_ids'])) != 6 or not set(cfg['development_case_ids'])<=set(dev):
        raise ValueError('观察必须使用固定六条新开发题，不用训练或最终测试题。')
    if not 1<=cfg['development_max_new_tokens']<=128:
        raise ValueError('开发生成长度不符。')
    files = [config, Path(__file__), ROOT/'preference_data.py', ROOT/'requirements-m6-lock.txt'] + [ROOT/cfg[k] for k in
             ['pair_file','pair_manifest','review_snapshot','candidate_snapshot','review_receipt','review_pack']]
    conditions = {'source_conditions':prep['conditions'],'files':{str(p.relative_to(ROOT)):m6.digest(p) for p in files},
                  'packages':m6.m6_versions(),'purpose':cfg['purpose'],'config':cfg}
    return cfg,rows,source_cfg,source_adapter,prompt,[dev[x] for x in cfg['development_case_ids']],conditions


def encoded_rows(tokenizer, rows, maximum):
    data, counts = [], []
    for row in rows:
        encoded = encode_preference(tokenizer,row['prompt'],row['chosen'],row['rejected'],maximum)
        data.append({k:encoded[k] for k in ('prompt','chosen','rejected')})
        counts.append({'id':row['id'],'prompt':len(encoded['prompt_input_ids']),
                       'chosen':len(encoded['chosen_input_ids']),'rejected':len(encoded['rejected_input_ids'])})
    return data,counts


def inspect(config=CONFIG):
    cfg,rows,source_cfg,adapter,prompt,dev,conditions=preflight(config)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
    data,counts=encoded_rows(tokenizer,rows,cfg['max_length'])
    selected=next(i for i,r in enumerate(rows) if r['id']=='m6-train-technical-02')
    row,count=rows[selected],counts[selected]
    print('这是实际数据编码检查，不加载Qwen、不生成回复、不更新LoRA。')
    print('样本：'+row['id']+'\n问题：'+row['prompt'][-1]['content'])
    print('\nchosen（'+('AI修订' if row['chosen_is_revision'] else '原始SFT候选')+'）：\n'+row['chosen'])
    print('\nrejected（原始SFT候选）：\n'+row['rejected'])
    print('\nToken：'+json.dumps(count,ensure_ascii=False))
    print('输入两次都相同；只累计最后回答＋一个EOS的log概率，问题/前文/padding不参加目标累计。')
    print('20对中直接选原候选2对，AI修订chosen18对。后续更新5次；本命令尚未训练。')


def trainer_class():
    from trl import DPOTrainer
    class FrozenSFTReferenceTrainer(DPOTrainer):
        def dpo_loss(self,chosen_logps,rejected_logps,ref_chosen_logps,ref_rejected_logps,loss_type='sigmoid',model_output=None):
            # 对比与logsigmoid使用FP32，保留梯度，避免BF16标量损失舍入。
            return super().dpo_loss(chosen_logps.float(),rejected_logps.float(),
                                    ref_chosen_logps.float(),ref_rejected_logps.float(),loss_type,model_output)
        @contextmanager
        def null_ref_context(self):
            # TRL 0.24.0的原上下文在异常时不保证切回；显式恢复策略。
            if not self.ref_adapter_name or not self.model_adapter_name:
                raise ValueError('本项目必须指定SFT reference与policy，不能默默禁用LoRA作参考。')
            model=self.accelerator.unwrap_model(self.model)
            try:
                model.set_adapter(self.ref_adapter_name)
                yield
            finally:
                model.set_adapter(self.model_adapter_name)
    return FrozenSFTReferenceTrainer


def copy_reference_from_policy(model):
    """只在优化前精确复制一次，避免第二次load_adapter的BF16→FP32舍入差。"""
    import torch
    named=dict(model.named_parameters())
    copied=0
    with torch.no_grad():
        for name,param in named.items():
            if '.policy.' not in name or 'lora_' not in name:continue
            reference=named[name.replace('.policy.','.reference.')]
            if reference.shape!=param.shape:raise ValueError('policy/reference形状不同。')
            reference.data=param.detach().clone()
            reference.requires_grad_(False)
            copied+=1
    if not copied:raise ValueError('没有可复制的SFT LoRA。')
    return copied


def make_model(source_cfg,adapter):
    from peft import PeftModel
    from scripts.m3 import load_base
    base,tokenizer,device=load_base(source_cfg)
    model=PeftModel.from_pretrained(base,adapter,adapter_name='policy',is_trainable=True,local_files_only=True)
    model.load_adapter(adapter,adapter_name='reference',is_trainable=False,local_files_only=True)
    model.set_adapter('policy')
    # 第一份policy先与磁盘SFT逐tensor核对；参考再精确复制，不放宽起点校验。
    from safetensors.torch import load_file
    saved=load_file(str(adapter/'adapter_model.safetensors'))
    for name,param in model.named_parameters():
        if '.policy.' in name and 'lora_' in name:
            key=name.replace('.policy.','.')
            if key not in saved or not param.detach().float().cpu().equal(saved[key].float()):
                raise ValueError('policy不是磁盘SFT的精确FP32参数。')
    copy_reference_from_policy(model)
    model.config.use_cache=False
    source=m6.read_json(adapter.parent/'run_manifest.json')
    if device!=source['device'] or str(next(model.parameters()).dtype)!=source['dtype']:
        raise ValueError('DPO设备/精度与SFT来源不同。')
    named=dict(model.named_parameters())
    policy={n:p for n,p in named.items() if '.policy.' in n and 'lora_' in n}
    if not policy or any(p.requires_grad and n not in policy for n,p in named.items()):
        raise ValueError('只能更新policy LoRA。')
    for name,param in policy.items():
        ref=named[name.replace('.policy.','.reference.')]
        if ref.requires_grad or not param.detach().equal(ref.detach()):
            raise ValueError('reference必须来自相同SFT且冻结。')
    print('已复制M3 SFT：policy LoRA可训练，reference LoRA冻结；基础权重共用。',flush=True)
    return model,tokenizer,device


def make_trainer(model,tokenizer,rows,cfg,output,callbacks=None):
    from datasets import Dataset
    from trl import DPOConfig
    data,counts=encoded_rows(tokenizer,rows,cfg['max_length'])
    args=DPOConfig(output_dir=str(output),max_steps=cfg['max_steps'],learning_rate=cfg['learning_rate'],
        beta=cfg['beta'],loss_type='sigmoid',per_device_train_batch_size=1,gradient_accumulation_steps=4,
        model_adapter_name='policy',ref_adapter_name='reference',precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,max_length=cfg['max_length'],max_prompt_length=None,max_completion_length=None,
        padding_free=False,disable_dropout=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',
        logging_steps=1,logging_first_step=True,logging_nan_inf_filter=False,report_to='none',
        eval_strategy='no',save_strategy='steps',save_steps=cfg['max_steps'],disable_tqdm=True,
        dataloader_pin_memory=False,dataloader_num_workers=0,seed=cfg['seed'],data_seed=cfg['seed'],
        bf16=False,fp16=False,remove_unused_columns=True,max_grad_norm=1.0)
    trainer=trainer_class()(model=model,args=args,processing_class=tokenizer,
        train_dataset=Dataset.from_list(data),callbacks=callbacks or [])
    for prepared,expected in zip(trainer.train_dataset,rows):
        correct=encode_preference(tokenizer,expected['prompt'],expected['chosen'],expected['rejected'],cfg['max_length'])
        for key in ('prompt_input_ids','chosen_input_ids','rejected_input_ids'):
            if prepared[key]!=correct[key]:
                raise ValueError('TRL实际编码与已核查边界/EOS不一致：'+key)
    return trainer,counts


def adapter_state(model,name):
    return {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if 'lora_' in n and '.'+name+'.' in n}


def base_probe(model,tokenizer,messages,device):
    import torch
    modes=[(module,module.training) for module in model.modules()]
    try:
        model.eval()
        with model.disable_adapter(),torch.no_grad():
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
            return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float().cpu().contiguous()
    finally:
        model.set_adapter('policy')
        for module,training in modes:module.training=training


def observe(model,tokenizer,device,prompt,cases,cfg,phase):
    import numpy as np
    import torch
    modes=[(m,m.training) for m in model.modules()]
    cpu_rng=torch.get_rng_state();mps_rng=torch.mps.get_rng_state() if device=='mps' else None
    python_rng=random.getstate();numpy_rng=np.random.get_state();cache=model.config.use_cache
    replies=[]
    try:
        model.set_adapter('policy');model.eval()
        for case in cases:
            messages=[{'role':'system','content':prompt}]+case['messages']
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
            with torch.inference_mode():
                output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                    temperature=1.,top_p=1.,top_k=50,max_new_tokens=cfg['development_max_new_tokens'],
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            new=output[0,ids.shape[1]:]
            answer=tokenizer.decode(new,skip_special_tokens=True).strip()
            replies.append({'id':case['id'],'messages':messages,'answer':answer,'output_tokens':new.numel(),
                            'finish_reason':'eos' if int(new[-1])==tokenizer.eos_token_id else 'length'})
            print(phase+'/'+case['id']+'：'+answer,flush=True)
        return replies
    finally:
        model.config.use_cache=cache
        for module,training in modes:module.training=training
        torch.set_rng_state(cpu_rng)
        if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
        random.setstate(python_rng);np.random.set_state(numpy_rng)
        model.set_adapter('policy')


def check_engine(config=CONFIG):
    import torch
    cfg,rows,source_cfg,adapter,prompt,dev,conditions=preflight(config)
    model,tokenizer,device=make_model(source_cfg,adapter)
    policy_before=adapter_state(model,'policy');ref_before=adapter_state(model,'reference')
    with tempfile.TemporaryDirectory(prefix='m6-dpo-forward-') as directory:
        _,all_counts=encoded_rows(tokenizer,rows,cfg['max_length'])
        longest=max(range(len(rows)),key=lambda i:all_counts[i]['prompt']+max(all_counts[i]['chosen'],all_counts[i]['rejected']))
        trainer,counts=make_trainer(model,tokenizer,[rows[0],rows[longest]],cfg,Path(directory))
        trainer.get_train_dataloader()  # 仅计算参考概率，未创建或执行优化步骤。
        batch=trainer.data_collator([trainer.train_dataset[0]])
        batch={k:v.to(device) for k,v in batch.items()}
        model.eval()
        with torch.no_grad():
            out=trainer.concatenated_forward(model,batch)
            rc,rr=trainer.compute_ref_log_probs(batch)
            pc,pr=out['chosen_logps'],out['rejected_logps']
            losses,_,_=trainer.dpo_loss(pc,pr,rc,rr)
        if not torch.allclose(pc,rc,rtol=1e-4,atol=1e-3) or not torch.allclose(pr,rr,rtol=1e-4,atol=1e-3):
            raise ValueError('起点policy与SFT reference概率不一致。')
        # 验证真实Trainer损失调用及最长样本反向传播，不调用optimizer.step。
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        longest_batch=trainer.data_collator([trainer.train_dataset[1]])
        longest_batch={k:v.to(device) for k,v in longest_batch.items()}
        backward_loss,_=trainer.get_batch_loss_metrics(model,longest_batch,train_eval='train')
        backward_loss.backward()
        gradients=[p.grad for n,p in model.named_parameters() if '.policy.' in n and p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise ValueError('策略梯度不存在或不有限。')
        if any(p.grad is not None for n,p in model.named_parameters() if '.policy.' not in n):raise ValueError('梯度泄漏到参考或基础参数。')
        gradient_norm=math.sqrt(sum(float(g.detach().float().pow(2).sum().cpu()) for g in gradients))
        if gradient_norm<=0:raise ValueError('策略梯度全零。')
        model.zero_grad(set_to_none=True)
        if not all(torch.equal(p,dict(model.named_parameters())[n].detach().float().cpu()) for n,p in {**policy_before,**ref_before}.items()):
            raise ValueError('前向检查不得更新Adapter。')
        report={'checked_at_utc':m6.now(),'purpose':'真实Qwen两对前向/参考及最长样本反向检查，无优化更新',
            'device':device,'dtype':str(next(model.parameters()).dtype),'initial_dpo_loss':float(losses.mean()),
            'policy_reference_equal':True,'adapter_unchanged':True,'optimizer_steps':0,'data_token_counts':counts,
            'backward_passes':1,'policy_gradient_norm':gradient_norm,'base_reference_gradients_absent':True,
            'memory_allocated_bytes':torch.mps.current_allocated_memory() if device=='mps' else None,
            'source_conditions':conditions}
        m6.dump_json(ROOT/'results/m6_setup/dpo_engine_check.json',report)
        print('真实Qwen DPO前向/反向检查通过；初始loss='+str(report['initial_dpo_loss'])+'；优化更新0次。')


def training_plan(cfg,conditions,config,run,result):
    plan=result/'plan.json'
    if plan.exists():
        if m6.read_json(plan)['conditions']!=conditions:
            raise ValueError('学习run输入或代码已变化，需恢复原版本或新run，不覆盖证据。')
        return
    run.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
    for relative in conditions['files']:
        target=run/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    m6.dump_json(plan,{'prepared_at_utc':m6.now(),'conditions':conditions,'formal_quality_experiment':False})


def train(config=CONFIG):
    import torch
    from transformers import TrainerCallback
    from safetensors.torch import save_file
    cfg,rows,source_cfg,adapter,prompt,dev,conditions=preflight(config)
    run,result=paths(cfg)
    training_plan(cfg,conditions,config,run,result)
    if (run/'run_manifest.json').exists():
        record=m6.read_json(run/'run_manifest.json')
        for filename,expected in record['adapter_files'].items():
            if m6.digest(run/'adapter/policy'/filename)!=expected:raise ValueError('已保存Adapter已变化。')
        print('此学习run已完成；没有重复训练。另开进程执行verify。');return
    if (run/'started.json').exists():
        raise ValueError('此短验证run已有未完成记录；保留失败产物，另建run配置重做。本入口不支持未经验证的checkpoint恢复。')
    with m6.run_lock():
        m6.dump_json(run/'started.json',{'started_at_utc':m6.now(),'training_process_pid':os.getpid()})
        progress={'phase':'loading','step':0,'max_steps':5,'purpose':cfg['purpose']}
        m6.dump_json(result/'progress.json',progress)
        try:
            model,tokenizer,device=make_model(source_cfg,adapter)
            policy_before=adapter_state(model,'policy');ref_before=adapter_state(model,'reference')
            original_base=base_probe(model,tokenizer,rows[0]['prompt'],device)
            memory=[]
            class Progress(TrainerCallback):
                def on_step_end(self,args,state,control,**kwargs):
                    progress.update(phase='training',step=state.global_step,updated_at_utc=m6.now())
                    m6.dump_json(result/'progress.json',progress)
                    if device=='mps':memory.append({'step':state.global_step,'allocated_bytes':torch.mps.current_allocated_memory(),
                                                  'driver_allocated_bytes':torch.mps.driver_allocated_memory()})
                def on_log(self,args,state,control,logs=None,**kwargs):
                    for k,v in (logs or {}).items():
                        if isinstance(v,(int,float)) and not math.isfinite(v):raise RuntimeError('非有限训练指标：'+k)
                    m6.dump_json(result/'training_log.json',state.log_history)
            trainer,counts=make_trainer(model,tokenizer,rows,cfg,run/'trainer',[Progress()])
            before=observe(model,tokenizer,device,prompt,dev,cfg,'DPO前')
            m6.dump_json(result/'development_timeline.json',{'purpose':'开发观察，非最终质量结论','before':before,'after':[]})
            trainer.get_train_dataloader()
            cache={'pair_ids':[r['id'] for r in rows],'conditions_sha256':object_digest(conditions),
                   'reference_is':'冻结的m3_learning_01 SFT，不是禁用Adapter的基模',
                   'chosen':list(trainer.train_dataset['ref_chosen_logps']),
                   'rejected':list(trainer.train_dataset['ref_rejected_logps'])}
            m6.dump_json(run/'reference_probabilities.json',cache)
            print('参考概率已固定；现在trainer.train才开始更新policy LoRA。',flush=True)
            started=time.perf_counter();out=trainer.train();elapsed=time.perf_counter()-started
            model.set_adapter('policy')
            after_policy=adapter_state(model,'policy');after_ref=adapter_state(model,'reference')
            changed=sum(not torch.equal(policy_before[n],p) for n,p in after_policy.items())
            if not changed or any(not torch.isfinite(p).all() for p in after_policy.values()):raise ValueError('策略未更新或不有限。')
            if any(not torch.equal(ref_before[n],p) for n,p in after_ref.items()):raise ValueError('参考LoRA发生变化。')
            final_base=base_probe(model,tokenizer,rows[0]['prompt'],device)
            if not torch.allclose(original_base,final_base,rtol=1e-3,atol=1e-2):raise ValueError('基础预测发生变化。')
            after=observe(model,tokenizer,device,prompt,dev,cfg,'DPO后')
            model.eval();model.config.use_cache=True
            ids=tokenizer.apply_chat_template(rows[0]['prompt'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
            with torch.no_grad():probe=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float().cpu().contiguous()
            model.save_pretrained(run/'adapter',selected_adapters=['policy'],save_embedding_layers=False)
            saved=run/'adapter/policy';tokenizer.save_pretrained(saved)
            save_file({'policy_logits':probe},str(run/'reload_probe.safetensors'))
            timeline={'purpose':'20对学习验证的开发观察，不宣称质量改善','before':before,'after':after}
            m6.dump_json(result/'development_timeline.json',timeline)
            lines=['# M6小样本DPO开发观察','','仅5次更新，非最终评测；每题使用固定short与同一前文。','']
            for left,right in zip(before,after):
                lines+=['## '+left['id'],'','用户：'+left['messages'][-1]['content'],'','### DPO前','',left['answer'],'','### DPO后','',right['answer'],'']
            (result/'development_timeline.md').write_text('\n'.join(lines),encoding='utf-8')
            if preflight(config)[-1]!=conditions:raise ValueError('训练期间来源条件改变。')
            record={'completed_at_utc':m6.now(),'training_process_pid':os.getpid(),'purpose':cfg['purpose'],'conditions':conditions,
                'device':device,'dtype':str(next(model.parameters()).dtype),'training_examples':20,'global_step':trainer.state.global_step,
                'changed_policy_lora_tensors':changed,'reference_unchanged':True,'base_probe_unchanged':True,
                'source_sft_unchanged':True,'initialization':'policy/reference均复制m3_learning_01；保存的policy为完整训练后LoRA，不叠加旧Adapter',
                'training_seconds':round(elapsed,3),'metrics':out.metrics,'data_token_counts':counts,'memory_samples':memory,
                'memory_note':'MPS周期采样，不是已证明的峰值内存','verification_messages':rows[0]['prompt'],
                'adapter_files':{p.name:m6.digest(p) for p in saved.iterdir() if p.is_file()},
                'reference_cache_sha256':m6.digest(run/'reference_probabilities.json'),'reload_probe_sha256':m6.digest(run/'reload_probe.safetensors'),
                'm6_final_test_generated':False,'formal_m6_quality_experiment_complete':False}
            m6.dump_json(run/'run_manifest.json',record);m6.dump_json(result/'training.json',record)
            progress.update(phase='trained_saved',step=trainer.state.global_step);m6.dump_json(result/'progress.json',progress)
            print('20对DPO学习验证训练已保存；下一步在新进程verify。')
        except BaseException as error:
            progress.update(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),updated_at_utc=m6.now())
            m6.dump_json(result/'progress.json',progress)
            raise


def verify(config=CONFIG):
    import torch
    from peft import PeftModel
    from scripts.m3 import load_base
    from safetensors.torch import load_file
    cfg,rows,source_cfg,source_adapter,prompt,dev,conditions=preflight(config)
    run,result=paths(cfg);record=m6.read_json(run/'run_manifest.json')
    if record['conditions']!=conditions or record['training_process_pid']==os.getpid():raise ValueError('需要同条件新进程重载。')
    for relative,expected in conditions['files'].items():
        if m6.digest(run/'source_snapshot'/relative)!=expected:raise ValueError('实际训练源码/数据快照已变化。')
    saved=run/'adapter/policy'
    for name,expected in record['adapter_files'].items():
        if m6.digest(saved/name)!=expected:raise ValueError('保存的policy Adapter已变化。')
    if m6.digest(run/'reload_probe.safetensors')!=record['reload_probe_sha256'] or \
       m6.digest(run/'reference_probabilities.json')!=record['reference_cache_sha256']:raise ValueError('重载探针或参考缓存已变化。')
    base,tokenizer,device=load_base(source_cfg)
    model=PeftModel.from_pretrained(base,saved,is_trainable=False,local_files_only=True);model.eval();model.requires_grad_(False)
    if device!=record['device'] or str(next(model.parameters()).dtype)!=record['dtype']:raise ValueError('重载设备/精度不同。')
    ids=tokenizer.apply_chat_template(record['verification_messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
    with torch.inference_mode():actual=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float().cpu()
    expected=load_file(str(run/'reload_probe.safetensors'))['policy_logits']
    error=float((actual-expected).abs().max())
    if not torch.allclose(actual,expected,rtol=1e-3,atol=1e-2):raise ValueError('新进程数值重载不一致。')
    m6.dump_json(result/'reload_verification.json',{'verified':True,'verified_at_utc':m6.now(),'new_process_pid':os.getpid(),
        'different_process':True,'max_logit_difference':error,'adapter_sha256':m6.digest(saved/'adapter_model.safetensors'),
        'run_manifest_sha256':m6.digest(run/'run_manifest.json'),'purpose':'仅确认20对小样本学习产物可重载，非完整M6质量结论'})
    print('小样本DPO新进程重载通过；最大logit差：'+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['inspect','check-engine','train','verify','status'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();config=args.config.resolve()
    try:
        if args.step=='status':
            cfg=preflight(config)[0];run,result=paths(cfg)
            if (run/'run_manifest.json').exists():print('20对DPO学习训练已完成；重载：'+str((result/'reload_verification.json').exists()))
            elif (result/'progress.json').exists():print(json.dumps(m6.read_json(result/'progress.json'),ensure_ascii=False))
            else:print('20对审核数据就绪；尚未启动Qwen DPO。inspect不训练，train才训练。')
        else:{'inspect':inspect,'check-engine':check_engine,'train':train,'verify':verify}[args.step](config)
    except KeyboardInterrupt:raise SystemExit('已中断，已写日志保留。本短验证不支持恢复，另建run后重做。')
    except (ValueError,RuntimeError,OSError,KeyError) as error:raise SystemExit('M6 DPO未通过：'+str(error))


if __name__=='__main__':main()
