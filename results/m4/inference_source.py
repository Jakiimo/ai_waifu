"""M4：先在终端调用reply，再接Gradio聊天界面。只推理，不训练。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / '.cache/gradio'))

from prepare_data import FROZEN, PROMPTS, TRAIN, digest, read_json, verify_frozen
from scripts.m1 import versions

DEFAULT_CONFIG = ROOT / 'configs/m4.json'


def validate_assets(config_path=DEFAULT_CONFIG):
    """确认用的是已完成且重载通过的这轮Adapter；不加载模型。"""
    cfg = read_json(config_path)
    for key in ('max_input_tokens', 'max_new_tokens'):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(key + '必须是正整数。')
    if cfg['do_sample'] is not False:
        raise ValueError('本轮Demo固定贪心解码，便于同题比较。')
    training_path = ROOT / cfg['training_config']
    train_cfg = read_json(training_path)
    from scripts.m3 import paths
    run, adapter, results = paths(train_cfg)
    record = read_json(run / 'run_manifest.json')
    verify_frozen()
    model_manifest = read_json(ROOT / 'models/model_manifest.json')
    expected = {
        'config_sha256': digest(training_path), 'train_sha256': digest(TRAIN),
        'm2_frozen_sha256': digest(FROZEN), 'model': model_manifest,
    }
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError('训练来源与当前文件不同：' + key)
    if record['packages'] != versions():
        raise ValueError('训练相关依赖版本已变化，请使用固定环境。')
    if digest(run / 'training_script.py') != record['training_script_sha256']:
        raise ValueError('实际训练源码快照已变化。')
    if digest(run / 'training_config.json') != record['config_sha256']:
        raise ValueError('实际训练配置快照已变化。')
    for name, expected_hash in record['adapter_files'].items():
        if digest(adapter / name) != expected_hash:
            raise ValueError('Adapter文件已变化：' + name)
    verification = read_json(results / 'reload_verification.json')
    if not verification['verified'] or not verification['different_process']:
        raise ValueError('这轮Adapter尚未通过独立进程重载验证。')
    if verification['adapter_sha256'] != digest(adapter / 'adapter_model.safetensors') or \
       verification['run_manifest_sha256'] != digest(run / 'run_manifest.json'):
        raise ValueError('重载验证属于其他训练产物。')
    prompts = read_json(PROMPTS)
    if cfg['chat_prompt'] != 'short' or cfg['comparison_prompt'] != 'strong_examples':
        raise ValueError('聊天使用short，同题比较使用冻结的strong_examples。')
    if cfg['max_input_tokens'] > 768 or cfg['max_new_tokens'] > 256:
        raise ValueError('本轮输入/生成上限不超过冻结实验设置。')
    return cfg, train_cfg, adapter, prompts


def prepare_messages(tokenizer, system_prompt, history, user_text, max_input_tokens):
    """保留角色设定、当前问题与最近完整轮次；不截断当前问题。"""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError('请先输入一句话。')
    user_text = user_text.strip()
    kept = [dict(message) for message in history]
    if len(kept) % 2:
        raise ValueError('会话历史必须包含完整问答轮次，请开始新会话。')
    for i, message in enumerate(kept):
        if message.get('role') != ('user' if i % 2 == 0 else 'assistant') or \
           not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('会话历史格式不正确，请开始新会话。')
    dropped = 0
    while True:
        messages = [{'role': 'system', 'content': system_prompt}] + kept + [
            {'role': 'user', 'content': user_text}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) <= max_input_tokens:
            return messages, kept, dropped, len(ids)
        if not kept:
            raise ValueError('这条消息太长，请缩短后重试；当前问题没有被截断。')
        kept = kept[2:]
        dropped += 1


class ChatEngine:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.cfg, self.train_cfg, self.adapter, self.prompts = validate_assets(config_path)
        self.model = self.tokenizer = self.device = None
        self.lock = threading.RLock()

    def ensure_loaded(self):
        """首次使用时加载一次；失败后允许下次重新加载。"""
        if self.model is not None:
            return
        from peft import PeftModel
        from scripts.m3 import load_base
        base, tokenizer, device = load_base(self.train_cfg)
        # 将磁盘中已经训练好的A/B加载进来，不创建新的训练任务。
        model = PeftModel.from_pretrained(base, self.adapter, is_trainable=False, local_files_only=True)
        model.eval()
        model.config.use_cache = True
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError('Demo中所有参数应冻结，只执行推理。')
        self.model, self.tokenizer, self.device = model, tokenizer, device

    def _generate(self, messages, mode):
        """原模型使用disable_adapter；LoRA模式使用同一份基础权重和A/B。"""
        import torch
        if mode not in ('base', 'lora'):
            raise ValueError('未知模型模式。')
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt').to(self.device)
        context = self.model.disable_adapter() if mode == 'base' else nullcontext()
        started = time.perf_counter()
        try:
            with context, torch.inference_mode():
                output = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=self.cfg['max_new_tokens'], do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            # PEFT退出disable_adapter时可能重新开启LoRA的requires_grad。
            # Demo没有优化器；仍显式恢复冻结状态，成功/异常两条路径都处理。
            self.model.requires_grad_(False)
        new_tokens = output[0, ids.shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer or not any(c.isalnum() for c in answer):
            raise RuntimeError('本次没有生成有效回复，请重试或开始新会话。')
        return {'answer': answer, 'mode': mode, 'input_tokens': ids.shape[1],
                'output_tokens': new_tokens.numel(),
                'finish_reason': 'eos' if int(new_tokens[-1]) == self.tokenizer.eos_token_id else 'length',
                'seconds': round(time.perf_counter() - started, 3)}

    def reply(self, user_text, history=None, mode='lora'):
        """文字+历史 → messages → token → generate → 文字+新历史。"""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError('请先输入一句话。')
        if mode not in ('base', 'lora'):
            raise ValueError('未知模型模式。')
        with self.lock:
            self.ensure_loaded()
            messages, kept, dropped, _ = prepare_messages(
                self.tokenizer, self.prompts[self.cfg['chat_prompt']], history or [],
                user_text, self.cfg['max_input_tokens'])
            result = self._generate(messages, mode)
            result.update(messages=messages, dropped_turns=dropped, history=kept + [
                {'role': 'user', 'content': user_text.strip()},
                {'role': 'assistant', 'content': result['answer']}])
            return result

    def compare(self, user_text):
        """比较区使用独立的同一输入，不借用聊天历史；顺序生成。"""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError('请先输入一个比较问题。')
        with self.lock:
            self.ensure_loaded()
            messages, _, _, _ = prepare_messages(
                self.tokenizer, self.prompts[self.cfg['comparison_prompt']], [],
                user_text, self.cfg['max_input_tokens'])
            base = self._generate(messages, 'base')
            adapted = self._generate(messages, 'lora')
            return {'messages': messages, 'base': base, 'lora': adapted,
                    'prompt_name': self.cfg['comparison_prompt'], 'do_sample': False}


def build_app(engine):
    import gradio as gr
    with gr.Blocks(title='凛音 · AI Waifu', analytics_enabled=False) as app:
        gr.Markdown('# 凛音\n嘴硬心软的聊天同伴。只记得当前会话里保留的对话；新会话后重新开始。')
        with gr.Tab('聊天'):
            mode = gr.Radio([('微调模型', 'lora'), ('原模型', 'base')], value='lora', label='当前模型')
            chatbot = gr.Chatbot(label='与凛音聊天', height=420)
            history = gr.State([])
            message = gr.Textbox(label='你的消息', placeholder='今天想聊什么？', lines=2)
            with gr.Row():
                send = gr.Button('发送', variant='primary')
                clear = gr.Button('新会话')
            status = gr.Markdown('模型尚未加载；首次发送时加载，请稍候。切换模型会开始新会话。')
        with gr.Tab('同题比较'):
            gr.Markdown('两边使用同一详细角色提示词、同一问题和生成设置。此区域独立于聊天会话。')
            compare_input = gr.Textbox(label='同一个问题', lines=2)
            compare_button = gr.Button('比较原模型与微调模型')
            with gr.Row():
                base_output = gr.Textbox(label='原模型', lines=8, interactive=False)
                lora_output = gr.Textbox(label='微调模型', lines=8, interactive=False)
            compare_status = gr.Markdown('等待输入；两边依次生成。')
        with gr.Tab('实验记录'):
            gr.Markdown('M3：360条实例、2轮、180次更新；开发loss 4.1188 → 3.3701。'
                        '\n\n已有回复仍存在训话、技术解释错误和指令偏离。口吻变化不代表整体质量提升。'
                        '\n\n完整记录见项目内 `results/m3_learning_01/development_comparison.md`；最终测试尚未执行。')

        controls = [message, send, mode, clear, compare_input, compare_button]

        def control_updates(enabled):
            return {component: gr.update(interactive=enabled) for component in controls}

        def send_chat(text, previous, selected):
            if not text or not text.strip():
                yield {status: '请先输入一句话。'}
                return
            yield {**control_updates(False), status: '正在加载或生成回复，请稍候……',
                   chatbot: previous + [{'role': 'user', 'content': text}]}
            try:
                result = engine.reply(text, previous, selected)
                notes = []
                if result['dropped_turns']:
                    notes.append(f'对话较长，已移除最早{result["dropped_turns"]}轮，保留最近对话。')
                if result['finish_reason'] == 'length':
                    notes.append('本条回复达到长度上限，可能未说完。')
                yield {**control_updates(True), history: result['history'], chatbot: result['history'],
                       message: gr.update(value='', interactive=True), status: '回复完成。' + ''.join(notes)}
            except Exception as error:
                print('M4回复失败：' + repr(error), flush=True)
                yield {**control_updates(True), chatbot: previous,
                       status: '回复失败，原对话已保留，可重试或开始新会话。原因：' + str(error)}

        def reset_chat(selected):
            label = '微调模型' if selected == 'lora' else '原模型'
            return {history: [], chatbot: [], message: '', status: f'已开始新会话，当前使用{label}。'}

        def compare_models(text):
            yield {**control_updates(False), base_output: '', lora_output: '',
                   compare_status: '正在依次生成两边的回答，请稍候……'}
            try:
                result = engine.compare(text)
                truncated = any(result[k]['finish_reason'] == 'length' for k in ('base', 'lora'))
                yield {**control_updates(True), base_output: result['base']['answer'],
                       lora_output: result['lora']['answer'],
                       compare_status: '比较完成。' + ('有回复达到长度上限，可能未说完。' if truncated else '')}
            except Exception as error:
                print('M4比较失败：' + repr(error), flush=True)
                yield {**control_updates(True), compare_status: '比较失败，可重试。原因：' + str(error)}

        # 发送按钮和Enter共用一个事件；防止同一生成任务被重复触发。
        gr.on([send.click, message.submit], send_chat, inputs=[message, history, mode],
              outputs=[history, chatbot, status] + controls, trigger_mode='once',
              concurrency_limit=1, concurrency_id='model', api_name='chat', show_progress='full')
        clear.click(reset_chat, inputs=mode, outputs=[history, chatbot, message, status],
                    concurrency_limit=1, concurrency_id='model', api_name=False)
        mode.change(reset_chat, inputs=mode, outputs=[history, chatbot, message, status],
                    concurrency_limit=1, concurrency_id='model', api_name=False)
        compare_button.click(compare_models, inputs=compare_input,
                             outputs=[base_output, lora_output, compare_status] + controls,
                             trigger_mode='once', concurrency_limit=1, concurrency_id='model',
                             api_name='compare', show_progress='full')
    return app.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--check', action='store_true', help='只核对模型产物和配置，不加载模型')
    parser.add_argument('--message', help='先在终端体验一次reply，不启动网页')
    parser.add_argument('--mode', choices=['base', 'lora'], default='lora')
    args = parser.parse_args()
    try:
        engine = ChatEngine(args.config)
        if args.check:
            print('M4模型产物检查通过；尚未加载模型或生成回答。')
            print('使用训练：' + engine.train_cfg['run_id'])
            print('Adapter：' + str(engine.adapter))
        elif args.message is not None:
            result = engine.reply(args.message, mode=args.mode)
            print('模型实际回复：\n' + result['answer'])
            print(f'输入{result["input_tokens"]} token；输出{result["output_tokens"]} token；{result["finish_reason"]}结束。')
        else:
            app = build_app(engine)
            app.launch(server_name=engine.cfg['server_name'], server_port=engine.cfg['server_port'],
                       share=False, inbrowser=False, show_error=True)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise SystemExit('M4未启动：' + str(error))


if __name__ == '__main__':
    main()
