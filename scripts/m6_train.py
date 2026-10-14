"""M6正式对照准备：同起点chosen-only SFT与DPO，训练由学习者启动。"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import m6,m6_dpo
from preference_data import encode_preference,object_digest,validate_decision

CONFIG=ROOT/'configs/m6_formal.json'


def paths(cfg,method):
    name=cfg['run_prefix']
    if not name.startswith('m6_formal_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError('run_prefix需要独立m6_formal_名称。')
    if method not in ('chosen_sft','dpo'):raise ValueError('未知训练方法。')
    name+='_'+method
    return ROOT/'outputs'/name,ROOT/'results'/name


def validate_rows(rows,split,cases,prompt,decisions,packs,raw):
    by_case={c['id']:c for c in cases};by_review={d['case_id']:d for d in decisions};seen=set()
    for row in rows:
        name=row['id']
        if name in seen or name not in by_case or name not in by_review:raise ValueError('重复/无来源偏好。')
        seen.add(name);case=by_case[name];review=by_review[name]
        if split not in ('train','dev') or row['split']!=split or case['split']!=split:raise ValueError('偏好划分不符。')
        if row['family_id']!=case['family_id'] or row['category']!=case['category'] or \
           row['prompt']!=[{'role':'system','content':prompt}]+case['messages'] or row['review']!=review:
            raise ValueError('输入/审核与固定场景不同。')
        if split=='dev' and (review['decision']=='revise' or row['chosen_is_revision']):raise ValueError('开发chosen不能修订凑量。')
        package=packs[review['pack_id']]
        item=next(i for i in package['items'] if i['case_id']==name)
        pair=sorted([r for r in raw if r['case_id']==name],key=lambda r:r['variant'])
        if len(pair)!=2 or object_digest(pair)!=item['candidates_sha256']:raise ValueError('实际候选版本不同。')
        accepted=validate_decision(review,item)
        if accepted is None or any(row[k]!=v for k,v in accepted.items()) or row['review_decision_sha256']!=object_digest(review):
            raise ValueError('偏好与实际审核不同。')


def preflight(config=CONFIG):
    cfg=m6.read_json(config);paths(cfg,'dpo')
    if cfg['num_train_epochs']!=1 or cfg['batch_size']!=1 or cfg['gradient_accumulation_steps']!=4 or \
       cfg['learning_rate']<=0 or cfg['beta']<=0 or cfg['observation_steps']!=15:
        raise ValueError('本轮固定1轮、batch1/累积4、每15步观察；另开run才变更方案。')
    data_cfg,cases,rubric,source_cfg,adapter,prompt,prep=m6.prepare()
    freeze=m6.read_json(ROOT/cfg['data_freeze'])
    if freeze['source_conditions']!=prep['conditions'] or freeze['test_generated'] or not freeze['training_inputs_ready']:
        raise ValueError('数据冻结来源或测试边界不符。')
    for name,expected in freeze['files'].items():
        if m6.digest(ROOT/name)!=expected:raise ValueError('冻结审核文件变化：'+name)
    train=m6.read_jsonl(ROOT/freeze['split_files']['train']);dev=m6.read_jsonl(ROOT/freeze['split_files']['dev'])
    if len(train)!=cfg['training_examples'] or len(dev)!=cfg['development_pairs'] or cfg['max_length']!=freeze['max_sequence_tokens']:
        raise ValueError('实际偏好数量或长度与配置不同。')
    decisions=m6.read_jsonl(ROOT/freeze['decision_snapshot'])
    raw=m6.read_candidates(ROOT/freeze['candidate_snapshot'],m6.all_tasks(cases,data_cfg,prompt),prep['conditions'])
    packs={p['pack_id']:p for p in (m6.read_json(ROOT/name) for name in freeze['pack_files'])}
    validate_rows(train,'train',cases,prompt,decisions,packs,raw);validate_rows(dev,'dev',cases,prompt,decisions,packs,raw)
    if {r['family_id'] for r in train}&{r['family_id'] for r in dev}:raise ValueError('训练开发场景族混用。')
    by_case={c['id']:c for c in cases if c['split']=='dev'}
    names=cfg['development_case_ids']
    if len(names)!=6 or len(set(names))!=6 or not set(names)<=set(by_case) or set(names)&{r['id'] for r in dev}:
        raise ValueError('生成探针需6条开发题，不能混入训练或开发偏好计算题。')
    files={**freeze['files'],str(config.relative_to(ROOT)):m6.digest(config),
           cfg['data_freeze']:m6.digest(ROOT/cfg['data_freeze']),
           'scripts/m6_train.py':m6.digest(Path(__file__)),'scripts/m6_dpo.py':m6.digest(ROOT/'scripts/m6_dpo.py'),
           'requirements-m6-lock.txt':m6.digest(ROOT/'requirements-m6-lock.txt')}
    conditions={'files':files,'source_conditions':prep['conditions'],'config':cfg,'packages':m6.m6_versions(),
                'selection':'固定1轮最终checkpoint，不使用最终测试或6组偏好选参'}
    return cfg,train,dev,source_cfg,adapter,prompt,[by_case[n] for n in names],conditions


def sft_rows(tokenizer,rows,maximum):
    data=[]
    for row in rows:
        pair=encode_preference(tokenizer,row['prompt'],row['chosen'],row['rejected'],maximum)
        prefix=pair['prompt_input_ids'];answer=pair['chosen_input_ids']
        ids=prefix+answer
        data.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prefix)+answer})
    return data


def make_trainer(method,model,tokenizer,train,dev,cfg,output,callbacks=None,force_cpu=False):
    from datasets import Dataset
    from transformers import Trainer,TrainingArguments,DataCollatorForSeq2Seq
    common=dict(output_dir=str(output),num_train_epochs=1,learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',
        eval_strategy='steps',eval_steps=15,save_strategy='steps',save_steps=15,save_total_limit=3,
        load_best_model_at_end=False,logging_steps=1,logging_nan_inf_filter=False,report_to='none',disable_tqdm=True,
        dataloader_pin_memory=False,dataloader_num_workers=0,seed=cfg['seed'],data_seed=cfg['seed'],
        bf16=False,fp16=False,max_grad_norm=1.0,use_cpu=force_cpu)
    # 两条路线都关闭dropout，保留冻结基础权重及同一个LoRA起点。
    from trl.trainer.utils import disable_dropout_in_model
    disable_dropout_in_model(model)
    if method=='chosen_sft':
        args=TrainingArguments(**common,label_names=['labels'])
        return Trainer(model=model,args=args,processing_class=tokenizer,
            train_dataset=Dataset.from_list(sft_rows(tokenizer,train,cfg['max_length'])),
            eval_dataset=Dataset.from_list(sft_rows(tokenizer,dev,cfg['max_length'])),
            data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100),callbacks=callbacks or [])
    from trl import DPOConfig
    args=DPOConfig(**common,beta=cfg['beta'],loss_type='sigmoid',model_adapter_name='policy',ref_adapter_name='reference',
        precompute_ref_log_probs=True,precompute_ref_batch_size=1,max_length=cfg['max_length'],
        max_prompt_length=None,max_completion_length=None,padding_free=False,disable_dropout=True)
    data,_=m6_dpo.encoded_rows(tokenizer,train,cfg['max_length']);eval_data,_=m6_dpo.encoded_rows(tokenizer,dev,cfg['max_length'])
    trainer=m6_dpo.trainer_class()(model=model,args=args,processing_class=tokenizer,
        train_dataset=Dataset.from_list(data),eval_dataset=Dataset.from_list(eval_data),callbacks=callbacks or [])
    for dataset,rows in [(trainer.train_dataset,train),(trainer.eval_dataset,dev)]:
        for actual,row in zip(dataset,rows):
            expected=encode_preference(tokenizer,row['prompt'],row['chosen'],row['rejected'],cfg['max_length'])
            if any(actual[k]!=expected[k] for k in ('prompt_input_ids','chosen_input_ids','rejected_input_ids')):
                raise ValueError('TRL编码边界不符。')
    return trainer


def inspect(config=CONFIG):
    cfg,train,dev,*_=preflight(config)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
    encoded=sft_rows(tokenizer,[train[0]],cfg['max_length'])[0]
    print('数据：179组train、6组dev；训练chosen有153条AI修订，开发无修订。')
    print('两条路线都从原M3 SFT开始，1轮约45次更新；固定最终checkpoint，未开始训练。')
    print('SFT只监督chosen；DPO同时比较chosen/rejected。相同步数不代表相同计算量。')
    print('样本：'+train[0]['id']+'\n问题：'+train[0]['prompt'][-1]['content'])
    print('SFT参与loss的文字：'+tokenizer.decode([x for x in encoded['labels'] if x!=-100]))
    print('DPO另使用的rejected：'+train[0]['rejected'])
    print('开发偏好6组覆盖不足，只作过程诊断；每15步另生成6条固定开发回答。')


def check_engine(method,config=CONFIG):
    import torch
    cfg,train,dev,source_cfg,adapter,prompt,cases,conditions=preflight(config)
    model,tokenizer,device=m6_dpo.make_model(source_cfg,adapter)
    before={**m6_dpo.adapter_state(model,'policy'),**m6_dpo.adapter_state(model,'reference')}
    _,counts=m6_dpo.encoded_rows(tokenizer,train,cfg['max_length'])
    index=max(range(len(train)),key=lambda i:counts[i]['prompt']+
              (counts[i]['chosen'] if method=='chosen_sft' else max(counts[i]['chosen'],counts[i]['rejected'])))
    with tempfile.TemporaryDirectory(prefix='m6-formal-qa-') as folder:
        trainer=make_trainer(method,model,tokenizer,[train[index]],dev[:1],cfg,Path(folder))
        if method=='dpo':trainer.get_train_dataloader()
        batch=trainer.data_collator([trainer.train_dataset[0]]);batch={k:v.to(device) for k,v in batch.items()}
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        loss=trainer.compute_loss(model,batch);loss.backward()
        grads=[p.grad for n,p in model.named_parameters() if '.policy.' in n and p.grad is not None]
        if not grads or not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):raise ValueError('梯度/loss不有限。')
        grad_norm=math.sqrt(sum(float(g.detach().float().pow(2).sum().cpu()) for g in grads))
        if grad_norm<=0:raise ValueError('梯度全零。')
        if any(p.grad is not None for n,p in model.named_parameters() if '.policy.' not in n):raise ValueError('梯度泄漏。')
        model.zero_grad(set_to_none=True)
        after={**m6_dpo.adapter_state(model,'policy'),**m6_dpo.adapter_state(model,'reference')}
        if any(not torch.equal(p,after[n]) for n,p in before.items()):raise ValueError('QA不得更新参数。')
        m6.dump_json(ROOT/f'results/m6_setup/formal_{method}_engine_check.json',{'method':method,'optimizer_steps':0,
            'backward_passes':1,'finite_policy_gradients':True,'base_reference_gradients_absent':True,
            'adapter_unchanged':True,'loss':float(loss.detach().cpu()),'policy_gradient_norm':grad_norm,'longest_case':train[index]['id'],
            'device':device,'conditions':conditions,'checked_at_utc':m6.now()})
    print(method+'真实模型最长样本反向检查通过；参数更新0次。')


def train(method,config=CONFIG):
    import torch
    from transformers import TrainerCallback
    from safetensors.torch import save_file
    cfg,rows,dev,source_cfg,adapter,prompt,cases,conditions=preflight(config)
    run,result=paths(cfg,method);run.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
    plan=result/'plan.json'
    if plan.exists():
        if m6.read_json(plan)['conditions']!=conditions:raise ValueError('已有run条件变化，需新run。')
    else:
        for relative in conditions['files']:
            p=run/'source_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,p)
        m6.dump_json(plan,{'conditions':conditions,'method':method,'prepared_at_utc':m6.now()})
    if (run/'run_manifest.json').exists():
        record=m6.read_json(run/'run_manifest.json')
        for name,sha in record['adapter_files'].items():
            if m6.digest(run/'adapter/policy'/name)!=sha:raise ValueError('已完成Adapter变化。')
        print('本run已完成，未重复训练；下一步新进程verify。');return
    if (run/'started.json').exists():raise ValueError('有未完成记录；保留产物，另开run配置重做。本入口不支持checkpoint恢复。')
    with m6.run_lock():
        m6.dump_json(run/'started.json',{'process_pid':os.getpid(),'started_at_utc':m6.now()})
        progress={'phase':'loading','step':0,'method':method};m6.dump_json(result/'progress.json',progress)
        timeline={'purpose':'固定开发生成观察，非最终测试或选参','method':method,'observations':[]};memory=[]
        try:
            model,tokenizer,device=m6_dpo.make_model(source_cfg,adapter)
            policy_before=m6_dpo.adapter_state(model,'policy');reference_before=m6_dpo.adapter_state(model,'reference')
            original_base=m6_dpo.base_probe(model,tokenizer,rows[0]['prompt'],device)
            expected_steps=math.ceil(len(rows)/4)
            def observe(step):
                answers=m6_dpo.observe(model,tokenizer,device,prompt,cases,cfg,method+'/step-'+str(step))
                timeline['observations'].append({'step':step,'answers':answers});m6.dump_json(result/'development_timeline.json',timeline)
                lines=['# M6 '+method+'开发观察','','固定方案、非最终评测；每15步及结束观察，问题不参与训练。','']
                for obs in timeline['observations']:
                    lines+=['## step '+str(obs['step']),'']
                    for a in obs['answers']:lines+=['### '+a['id'],'','用户：'+a['messages'][-1]['content'],'',a['answer'],'']
                (result/'development_timeline.md').write_text('\n'.join(lines),encoding='utf-8')
            class Progress(TrainerCallback):
                def on_step_end(self,args,state,control,**kwargs):
                    progress.update(phase='training',step=state.global_step,max_steps=state.max_steps)
                    m6.dump_json(result/'progress.json',progress)
                    if device=='mps':memory.append({'step':state.global_step,'allocated_bytes':torch.mps.current_allocated_memory(),
                        'driver_allocated_bytes':torch.mps.driver_allocated_memory()})
                    if state.global_step%15==0 or state.global_step==state.max_steps:observe(state.global_step)
                def on_log(self,args,state,control,logs=None,**kwargs):
                    if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in (logs or {}).values()):raise RuntimeError('指标非有限。')
                    m6.dump_json(result/'training_log.json',state.log_history)
            trainer=make_trainer(method,model,tokenizer,rows,dev,cfg,run/'trainer',[Progress()])
            observe(0)
            if method=='dpo':
                trainer.get_train_dataloader()
                cache={'conditions_sha256':object_digest(conditions),'pair_ids':[r['id'] for r in rows],
                    'reference':'冻结原M3 SFT','chosen':list(trainer.train_dataset['ref_chosen_logps']),
                    'rejected':list(trainer.train_dataset['ref_rejected_logps'])}
                m6.dump_json(run/'reference_probabilities.json',cache)
            print('现在开始'+method+'参数更新；原M3 SFT起点，预计45步。',flush=True)
            started=time.perf_counter();out=trainer.train();elapsed=time.perf_counter()-started
            if trainer.state.global_step!=expected_steps:raise ValueError('实际更新次数与单轮预算不同。')
            model.set_adapter('policy');policy_after=m6_dpo.adapter_state(model,'policy');ref_after=m6_dpo.adapter_state(model,'reference')
            changed=sum(not torch.equal(p,policy_after[n]) for n,p in policy_before.items())
            if not changed or any(not torch.isfinite(p).all() for p in policy_after.values()):raise ValueError('policy未更新/不有限。')
            if any(not torch.equal(p,ref_after[n]) for n,p in reference_before.items()):raise ValueError('reference变化。')
            if not torch.allclose(original_base,m6_dpo.base_probe(model,tokenizer,rows[0]['prompt'],device),rtol=1e-3,atol=1e-2):
                raise ValueError('基础预测探针变化。')
            model.eval();model.config.use_cache=True
            ids=tokenizer.apply_chat_template(rows[0]['prompt'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
            with torch.no_grad():probe=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float().cpu().contiguous()
            model.save_pretrained(run/'adapter',selected_adapters=['policy'],save_embedding_layers=False)
            saved=run/'adapter/policy';tokenizer.save_pretrained(saved);save_file({'policy_logits':probe},str(run/'reload_probe.safetensors'))
            if preflight(config)[-1]!=conditions:raise ValueError('训练期间输入/代码条件变化。')
            _,counts=m6_dpo.encoded_rows(tokenizer,rows,cfg['max_length'])
            record={'method':method,'conditions':conditions,'completed_at_utc':m6.now(),'training_process_pid':os.getpid(),
                'device':device,'dtype':str(next(model.parameters()).dtype),'global_step':trainer.state.global_step,
                'training_examples':len(rows),'development_pairs':len(dev),'training_seconds':round(elapsed,3),'metrics':out.metrics,
                'changed_policy_lora_tensors':changed,'reference_unchanged':True,'base_probe_unchanged':True,'source_sft_unchanged':True,
                'data_token_counts':counts,'answer_token_exposure':sum(c['chosen']+(c['rejected'] if method=='dpo' else 0) for c in counts),
                'budget_note':'1轮179组、45次更新；最后累积组3个样本。DPO两回答＋参考计算，不是等算力。',
                'memory_samples':memory,'memory_note':'周期采样，未声称峰值','verification_messages':rows[0]['prompt'],
                'adapter_files':{p.name:m6.digest(p) for p in saved.iterdir() if p.is_file()},
                'reload_probe_sha256':m6.digest(run/'reload_probe.safetensors'),'formal_quality_experiment_complete':False,
                'reference_cache_sha256':m6.digest(run/'reference_probabilities.json') if method=='dpo' else None}
            m6.dump_json(run/'run_manifest.json',record);m6.dump_json(result/'training.json',record)
            progress.update(phase='trained_saved',step=trainer.state.global_step);m6.dump_json(result/'progress.json',progress)
            print(method+'已保存；下一步新进程verify，不代表质量改善。')
        except BaseException as e:
            progress.update(phase='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e))
            m6.dump_json(result/'progress.json',progress);raise


def verify(method,config=CONFIG):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from scripts.m3 import load_base
    cfg,train,dev,source_cfg,adapter,prompt,cases,conditions=preflight(config);run,result=paths(cfg,method)
    record=m6.read_json(run/'run_manifest.json')
    if record['conditions']!=conditions or record['method']!=method or record['training_process_pid']==os.getpid():raise ValueError('需同条件新进程重载。')
    for name,sha in conditions['files'].items():
        if m6.digest(run/'source_snapshot'/name)!=sha:raise ValueError('训练快照变化。')
    for name,sha in record['adapter_files'].items():
        if m6.digest(run/'adapter/policy'/name)!=sha:raise ValueError('Adapter变化。')
    if m6.digest(run/'reload_probe.safetensors')!=record['reload_probe_sha256']:raise ValueError('探针变化。')
    if method=='dpo' and m6.digest(run/'reference_probabilities.json')!=record['reference_cache_sha256']:raise ValueError('参考缓存变化。')
    base,tokenizer,device=load_base(source_cfg)
    model=PeftModel.from_pretrained(base,run/'adapter/policy',is_trainable=False,local_files_only=True);model.eval();model.requires_grad_(False)
    if device!=record['device'] or str(next(model.parameters()).dtype)!=record['dtype']:raise ValueError('重载设备/精度不同。')
    ids=tokenizer.apply_chat_template(record['verification_messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
    with torch.inference_mode():actual=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float().cpu()
    expected=load_file(str(run/'reload_probe.safetensors'))['policy_logits'];error=float((actual-expected).abs().max())
    if not torch.allclose(actual,expected,rtol=1e-3,atol=1e-2):raise ValueError('新进程数值重载不一致。')
    m6.dump_json(result/'reload_verification.json',{'verified':True,'method':method,'new_process_pid':os.getpid(),
        'different_process':True,'max_logit_difference':error,'verified_at_utc':m6.now(),
        'run_manifest_sha256':m6.digest(run/'run_manifest.json'),'adapter_sha256':m6.digest(run/'adapter/policy/adapter_model.safetensors')})
    print(method+'新进程重载通过；最大logit差：'+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['inspect','check-engine','train','verify','status'])
    parser.add_argument('--method',choices=['chosen_sft','dpo'],default='chosen_sft')
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();config=args.config.resolve()
    try:
        if args.step=='status':
            cfg=preflight(config)[0]
            for method in ['chosen_sft','dpo']:
                run,result=paths(cfg,method)
                print(method+'：训练'+('完成' if (run/'run_manifest.json').exists() else '未完成')+'；重载'+str((result/'reload_verification.json').exists()))
        elif args.step=='inspect':inspect(config)
        else:globals()[args.step.replace('-','_')](args.method,config)
    except (ValueError,RuntimeError,OSError,KeyError,StopIteration) as e:raise SystemExit('M6对照未通过：'+str(e))


if __name__=='__main__':main()
