"""M6偏好数据契约与编码；不产生模型回答，不替AI或人类作偏好判断。"""
from collections import Counter
import hashlib
import json


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate_scenarios(rows, cfg):
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('场景id重复。')
    expected = {'daily', 'comfort', 'study', 'technical', 'stability', 'relationship'}
    families = {}
    for row in rows:
        if row['split'] not in cfg['planned_counts'] or row['category'] not in expected:
            raise ValueError('未知split/category：' + row['id'])
        if row['data_version'] != cfg['data_version'] or row['rubric_version'] != cfg['rubric_version']:
            raise ValueError('场景版本不同。')
        family = row['family_id']
        if family in families and families[family] != row['split']:
            raise ValueError('同一场景族跨集合：' + family)
        families[family] = row['split']
        messages = row['messages']
        if not messages or len(messages) % 2 != 1:
            raise ValueError('场景必须以user问题结束。')
        for i, message in enumerate(messages):
            if message.get('role') != ('user' if i % 2 == 0 else 'assistant') or \
               not isinstance(message.get('content'), str) or not message['content'].strip():
                raise ValueError('场景消息顺序或内容错误。')
        if 'chosen' in row or 'rejected' in row:
            raise ValueError('场景计划不能冒充已审核的偏好对。')
    for split, count in cfg['planned_counts'].items():
        actual = Counter(row['category'] for row in rows if row['split'] == split)
        if count % 6 or dict(actual) != {category: count // 6 for category in expected}:
            raise ValueError('类别数量不符：' + split)
    return {'cases': len(rows), 'families': len(families),
            'counts': dict(Counter(row['split'] for row in rows)),
            'contextual': sum(len(row['messages']) > 1 for row in rows)}


def build_candidate_tasks(cases, cfg, system_prompt, split, limit):
    if split not in ('train', 'dev'):
        raise ValueError('最终测试禁止生成训练候选。')
    selected = [case for case in cases if case['split'] == split]
    if not isinstance(limit, int) or not 1 <= limit <= len(selected):
        raise ValueError('limit表示本次取前多少个场景，必须在1与该split总数之间。')
    all_indices = {case['id']: index for index, case in enumerate(cases)}
    tasks = []
    for case in selected[:limit]:
        for variant in range(cfg['candidates_per_case']):
            tasks.append({'key': case['id'] + f':candidate-{variant}', 'case_id': case['id'],
                          'split': split, 'category': case['category'], 'variant': variant,
                          'seed': cfg['seed'] + all_indices[case['id']] * 100 + variant,
                          'messages': [{'role': 'system', 'content': system_prompt}] + case['messages']})
    # expectation与未来chosen/rejected绝不进入模型输入。
    return tasks


def encode_preference(tokenizer, messages, chosen, rejected, max_length):
    if not chosen.strip() or not rejected.strip() or chosen.strip() == rejected.strip():
        raise ValueError('偏好答案为空或完全相同。')
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    encoded = {'prompt': prefix, 'chosen': chosen, 'rejected': rejected, 'prompt_input_ids': prompt_ids}
    for key, answer in [('chosen', chosen), ('rejected', rejected)]:
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if tokenizer.eos_token_id in answer_ids:
            raise ValueError('答案含内部EOS，不把特殊结束标记当普通台词。')
        joined = tokenizer(prefix + answer, add_special_tokens=False)['input_ids']
        if joined != prompt_ids + answer_ids:
            raise ValueError('分开编码与完整序列边界不一致。')
        completion = answer_ids + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(completion) > max_length:
            raise ValueError('偏好序列超长，拒绝截断关键差异。')
        encoded[key + '_input_ids'] = completion
    return encoded


def completion_log_probability(logits, input_ids, answer_mask):
    """课堂用：因果预测错开一格，只累加目标回答token的log概率。"""
    import torch
    if logits.shape[:2] != input_ids.shape or answer_mask.shape != input_ids.shape:
        raise ValueError('logits、input_ids与目标mask形状不符。')
    targets = input_ids[:, 1:]
    mask = answer_mask[:, 1:].bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError('每条序列必须有目标回答。')
    log_probs = logits[:, :-1].float().log_softmax(-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, token_log_probs, 0).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """课堂用sigmoid DPO公式；正式训练仍复用TRL。"""
    import torch.nn.functional as F
    if beta <= 0:
        raise ValueError('beta必须为正。')
    margin = beta * ((policy_chosen - policy_rejected) -
                     (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin), margin


def validate_decision(decision, item):
    if decision['case_id'] != item['case_id'] or decision['candidates_sha256'] != item['candidates_sha256']:
        raise ValueError('审核属于其他场景或候选版本。')
    action = decision.get('decision')
    if action not in ('accept', 'revise', 'reject'):
        raise ValueError('审核decision必须accept/revise/reject。')
    if not decision.get('reason', '').strip() or not decision.get('reviewer', '').strip():
        raise ValueError('审核必须记录实际评分者和理由。')
    if action == 'reject':
        return None
    if not item['reviewable']:
        raise ValueError('生成失败或重复回答不能直接转成偏好对。')
    if decision.get('confidence') not in ('high', 'medium'):
        raise ValueError('低置信偏好不能入训。')
    if decision.get('preference') not in ('left', 'right'):
        raise ValueError('平局不作为chosen/rejected。')
    if decision.get('swap_consistent') is not True:
        raise ValueError('未完成交换左右一致性复核。')
    if item['category'] == 'technical' and decision.get('technical_verified') is not True:
        raise ValueError('技术题需实际核实，不能直接接受Judge猜测。')
    for key in ('chosen_voice', 'chosen_persona_context', 'chosen_usefulness'):
        if type(decision.get(key)) is not int or not 1 <= decision[key] <= 5:
            raise ValueError('chosen的三维评分无效：' + key)
    if decision['chosen_persona_context'] < 4 or decision['chosen_usefulness'] < 4:
        raise ValueError('chosen必须人设边界和有用性合格，不能只选较不差。')
    side = decision['preference']
    other = 'right' if side == 'left' else 'left'
    chosen = item[side + '_answer']
    if action == 'revise':
        chosen = decision.get('revised_chosen', '').strip()
        if not chosen or not decision.get('revision_source', '').strip():
            raise ValueError('修订chosen需文本和实际修订来源。')
    rejected = item[other + '_answer']
    if chosen.strip() == rejected.strip():
        raise ValueError('偏好两侧不能相同。')
    return {'chosen': chosen, 'rejected': rejected, 'chosen_side': side,
            'chosen_is_revision': action == 'revise'}
