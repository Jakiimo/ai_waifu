"""M2 单轮/多轮共用编码：只监督当前最后一句 assistant。"""

def encode_conversation(tokenizer, system_prompt, messages, max_length):
    if len(messages) < 2 or messages[-1]['role'] != 'assistant':
        raise ValueError('训练对话必须以 assistant 目标回答结束。')
    for i, message in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        if message.get('role') != expected or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('消息顺序错误或内容为空。')
    full = [{'role': 'system', 'content': system_prompt}] + messages
    prefix = tokenizer.apply_chat_template(full[:-1], tokenize=True, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError('目标回答的模板起点错误。')
    if len(ids) > max_length:
        raise ValueError(f'对话长度 {len(ids)} 超过 {max_length}，拒绝截断。')
    padding = max_length - len(ids)
    labels = [-100] * len(prefix) + ids[len(prefix):] + [-100] * padding
    return {'input_ids': ids + [tokenizer.pad_token_id] * padding,
            'attention_mask': [1] * len(ids) + [0] * padding, 'labels': labels}, {
                'prompt_tokens': len(prefix), 'answer_tokens': len(ids) - len(prefix), 'padding_tokens': padding}
