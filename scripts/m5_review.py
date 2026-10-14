"""M5匿名评分网页：只读取回答与保存人工选择，不加载或训练模型。"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / '.cache/gradio'))
import scripts.m5 as m5

EDITABLE = [field for field in m5.RATING_FIELDS if field not in ('pair_id', 'reviewable')]
REQUIRED = [field for field in EDITABLE if field != 'notes']


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != m5.RATING_FIELDS:
            raise ValueError('评分表列名已变化。')
        return list(reader)


def public_pairs(items):
    # 只供检查CSV格式/分数使用，不包含A/B/C/D映射。
    pairs = [{'pair_id': item['pair_id'], 'case_id': item['pair_id'], 'reviewable': item['reviewable']} for item in items]
    cases = [{'id': item['pair_id'], 'category': item['category']} for item in items]
    return pairs, cases


def save_rating(path, items, pair_id, values):
    items_by_id = {item['pair_id']: item for item in items}
    item = items_by_id[pair_id]
    rows = read_rows(path)
    if item['reviewable']:
        if len(values) != len(EDITABLE):
            raise ValueError('评分字段数量不正确。')
        update = {field: '' if value is None else str(value).strip() for field, value in zip(EDITABLE, values)}
        if any(not update[field] for field in REQUIRED):
            raise ValueError('请填完两边分数、错误标记与偏好；备注可以不填。')
        row = next(row for row in rows if row['pair_id'] == pair_id)
        row.update(update)
    pairs, cases = public_pairs(items)
    temp = path.with_name('ratings.pending.csv')
    try:
        m5.write_csv(temp, rows)
        # 使用同一个评分校验器，避免网页允许的数据被report拒绝。
        rated, pending = m5.parse_ratings(temp, pairs, cases)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return len(rated), pending


def build_app(review):
    import gradio as gr
    items = m5.read_json(review / 'review_items.json')
    path = review / 'ratings.csv'
    pairs, cases = public_pairs(items)
    m5.parse_ratings(path, pairs, cases)
    by_id = {item['pair_id']: item for item in items}
    ids = [item['pair_id'] for item in items]

    def view(pair_id):
        item = by_id[pair_id]
        rows = {row['pair_id']: row for row in read_rows(path)}
        row = rows[pair_id]
        rated, pending = m5.parse_ratings(path, pairs, cases)
        text = f'### {pair_id} · {item["category"]}\n\n'
        for message in item['messages']:
            role = '用户' if message['role'] == 'user' else '预置角色前文'
            text += role + '：' + message['content'] + '\n\n'
        text += '检查重点：' + item['expectation']
        progress = f'已完整评分{len(rated)}对，待评分{len(pending)}对。'
        if not item['reviewable']:
            progress += '本对含生成失败，不评分，点击下一题跳过；失败仍会进入统计。'
        return item, row, text, progress

    def first_pending():
        _, pending = m5.parse_ratings(path, pairs, cases)
        return pending[0] if pending else ids[0]

    selected = first_pending()
    item, row, text, progress_text = view(selected)
    with gr.Blocks(title='M5 · 匿名人工评分', analytics_enabled=False) as app:
        gr.Markdown('# M5 匿名人工评分\n先看准确、有用和是否遵守要求，再看凛音风格。左右模型身份隐藏。'
                    '\n\n每题请点击“保存并下一题”；切换题目前先保存。可以关闭后继续，不自动填分。')
        progress = gr.Markdown(progress_text)
        current = gr.Dropdown(ids, value=selected, label='当前题目')
        question = gr.Markdown(text)
        with gr.Row():
            left = gr.Textbox(item['left_answer'], label='左侧回答', lines=6, interactive=False)
            right = gr.Textbox(item['right_answer'], label='右侧回答', lines=6, interactive=False)
        fields = {}
        with gr.Row():
            for side, label in [('left', '左侧'), ('right', '右侧')]:
                with gr.Column():
                    for name, description in [('style', '人物与语气'), ('usefulness', '准确、有用、符合要求')]:
                        key = side + '_' + name
                        fields[key] = gr.Radio(['1', '2', '3', '4', '5'], value=row[key] or None,
                                               label=label + description + '（1–5）', interactive=item['reviewable'])
                    for name, description in [('fact_error', '事实错误'), ('constraint_error', '违反用户明确要求')]:
                        key = side + '_' + name
                        fields[key] = gr.Radio([('有', 'yes'), ('没有', 'no'), ('不适用', 'na')],
                                               value=row[key] or None, label=label + description, interactive=item['reviewable'])
        fields['preference'] = gr.Radio([('左侧更好', 'left'), ('差不多', 'tie'), ('右侧更好', 'right')],
                                        value=row['preference'] or None, label='整体偏好', interactive=item['reviewable'])
        fields['notes'] = gr.Textbox(row['notes'], label='备注（可选：具体哪里好或不好）', lines=2, interactive=item['reviewable'])
        save = gr.Button('保存并下一题', variant='primary')
        status = gr.Markdown('等待你的评分。技术题的事实错误请明确选择有/没有。')
        with gr.Accordion('角色设定与1–5分评分说明', open=False):
            gr.Markdown((review / '评分说明.md').read_text())

        def show_pair(pair_id):
            item, row, text, progress_text = view(pair_id)
            return {question: text, left: item['left_answer'], right: item['right_answer'], progress: progress_text,
                    **{component: gr.update(value=row[key] or None, interactive=item['reviewable']) for key, component in fields.items()}}

        def resume_view():
            target = first_pending()
            return {current: target, **show_pair(target)}

        def save_and_next(pair_id, *values):
            try:
                rated, pending = save_rating(path, items, pair_id, values)
                if pending:
                    index = ids.index(pair_id)
                    order = ids[index+1:] + ids[:index+1]
                    target = next(key for key in order if key in pending)
                    return {current: target, status: f'{pair_id}已保存。', progress: f'已完整评分{rated}对，待评分{len(pending)}对。'}
                return {status: '全部可评审回答已评分。回终端运行report汇总；有生成失败时仍需查看失败记录。',
                        progress: f'已完整评分{rated}对，待评分0对。'}
            except (ValueError, KeyError, StopIteration) as error:
                return {status: '尚未保存：' + str(error)}

        current.change(show_pair, inputs=current, outputs=[question, left, right, progress] + list(fields.values()),
                       concurrency_limit=1, concurrency_id='review', api_name=False)
        save.click(save_and_next, inputs=[current] + [fields[key] for key in EDITABLE],
                   outputs=[current, status, progress], concurrency_limit=1, concurrency_id='review',
                   trigger_mode='once', api_name=False)
        # Blocks初始值在进程启动时固定；每次打开/刷新页面都重新读取已保存的CSV。
        app.load(resume_view, outputs=[current, question, left, right, progress] + list(fields.values()),
                 concurrency_limit=1, concurrency_id='review', api_name=False)
    return app.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=m5.CONFIG)
    parser.add_argument('--port', type=int, default=7862)
    args = parser.parse_args()
    try:
        cfg, _, _, _, _ = m5.prepare(args.config.resolve())
        folder, private = m5.paths(cfg)
        m5.check_blind_integrity(folder, private)
        build_app(folder / 'review').launch(server_name='127.0.0.1', server_port=args.port,
                                           share=False, inbrowser=False)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('评分网页未启动：' + str(error) + '；先完成generate和blind。')


if __name__ == '__main__':
    main()
