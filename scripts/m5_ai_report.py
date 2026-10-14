"""汇总已完成的 AI 评分并标明来源；不生成答案、不修改评分。"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import m5


def main():
    cfg = m5.read_json(m5.CONFIG)
    folder, private = m5.paths(cfg)
    review = folder / 'ai_review'
    metadata = m5.read_json(review / 'metadata.json')
    for relative, expected in metadata['locked_files'].items():
        if m5.digest(ROOT / relative) != expected:
            raise ValueError('AI评分记录已变化，需重新评审并记录版本：' + relative)
    m5.report(reviewer=metadata['reviewer'])
    summary = m5.read_json(folder / 'summary.json')
    generated_report = (folder / 'report.md').read_text(encoding='utf-8')
    # 原脚本的 human 指标键与文字是旧人工流程约定；保存原始汇总供复核。
    m5.dump_json(review / 'generated_summary.json', summary)
    (review / 'generated_report.md').write_text(generated_report, encoding='utf-8')
    summary['model_judge'] = summary.pop('human')
    summary['grading_method'] = metadata
    summary['human_reviewer_count'] = 0
    summary['limitations'] = [
        '由当前对话的Codex单一AI评分者逐对评审，不是人工盲评或独立模型复核。',
        '评分助手参与过数据和代码准备，已知实验设计，首次检查还看到了已有一对评分；可从语气猜测身份。评分结束前未主动解析身份映射或原始分组回答。',
        '小型合成测试集，主题可能与训练相近；一次贪心输出，不宣称统计显著或普遍改善。',
        '1–5分是有解释的主观序数判断，AI也可能判断错误；没有人类评分者一致性或复测证据。',
        '本次已使用最终测试题；后续调模型或Prompt需另建最终测试集。',
        '输入token仅表示输入长度，不能自动解释为费用或速度收益。',
    ]
    m5.dump_json(folder / 'summary.json', summary)
    prefix = (
        '# M5 评测报告（AI评分）\n\n'
        '按用户“评分部分你来负责”的要求，本轮改为AI评审：120对、240条回答均已评完，每对理由保存在review/ratings.csv。'
        '你无需继续逐题打标。已有1对评分已另存备份，本轮全部由AI统一重评。\n\n'
        '评分者为当前对话Codex；具体后端模型版本未获取，不虚构版本号。身份映射在评分锁定后才用于汇总，'
        '但助手熟悉数据和实验，也看过原有1对评分，因此不称为独立或完全盲评。\n\n'
        '机器摘要中的model_judge字段为AI分数；human_reviewer_count=0。本轮完成的是AI评测，未完成原计划的人工评测。\n\n'
    )
    body = generated_report.split('\n', 1)[1]
    body = body.split('## 局限', 1)[0]
    body = body.replace('四组都成功生成的独立场景', '四组都成功生成的冻结测试场景')
    body = body.replace('由作者写出，不自动生成“显著提升”结论。', '结合AI评分与案例讨论，不自动生成“显著提升”结论。')
    # 展示次序固定B/D，notes却来自随机左右；明示原左右，避免把批评归错组。
    pairs = {p['case_id']: p for p in m5.read_json(private / 'blind_key.json')['pairs']
             if p['comparison'] == 'BD'}
    for match in list(re.finditer(r'### (test-[^\n]+)\n([\s\S]*?)(?=\n### |\Z)', body))[::-1]:
        pair = pairs[match.group(1)]
        text = match.group(0).replace('评分者备注：',
            f'原匿名编号{pair["pair_id"]}；备注中的左={pair["left_group"]}、右={pair["right_group"]}。\n\n评分者备注：')
        body = body[:match.start()] + text + body[match.end():]
    body += '## 局限\n\n' + '\n'.join('- ' + note for note in summary['limitations']) + '\n'
    (folder / 'report.md').write_text(prefix + body, encoding='utf-8')
    print('已标明AI评分来源，保留原始汇总；人工评分人数为0。')


if __name__ == '__main__':
    main()
