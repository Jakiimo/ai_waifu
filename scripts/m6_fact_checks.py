"""审核用最小事实验证，不加载语言模型，不训练，不执行候选中的代码。"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run():
    import torch
    checks = {}
    def save(case, actual, expected, explanation=''):
        if actual != expected:
            raise AssertionError((case, actual, expected))
        checks[case] = {'actual':actual, 'expected':expected, 'passed':True, 'explanation':explanation}

    def train(n, actual, expected, note=''):
        save(f'm6-train-technical-{n:02d}', actual, expected, note)
    def dev(n, actual, expected, note=''):
        save(f'm6-dev-technical-{n:02d}', actual, expected, note)

    train(1,[len('猫狗'),len('猫狗'.encode('utf-8'))],[2,6])
    train(2,list(range(2,7,2)),[2,4,6])
    d={}; value=d.get('x',0);train(3,[value,d],[0,{}])
    train(4,[[bool(x) for x in [0,[],None]],[type(x).__name__ for x in [0,[],None]]],[[False,False,False],['int','list','NoneType']])
    b=[1];a=b;a.append(2);train(5,[a is b,b],[True,[1,2]])
    train(6,[list(x) for x in enumerate(['甲','乙'],start=1)],[[1,'甲'],[2,'乙']])
    train(7,['3'+'4',3+4],['34',7])
    def add(a,b):return a+b
    train(8,add(2,3),5)
    def print_only():print('hi')
    capture=io.StringIO()
    with redirect_stdout(capture):returned=print_only()
    train(9,[returned,capture.getvalue()],[None,'hi\n'])
    train(10,[type((5,)).__name__,type((5)).__name__],['tuple','int'])
    broken=[];continued=[]
    for i in range(3):
        if i==1:break
        broken.append(i)
    for i in range(3):
        if i==1:continue
        continued.append(i)
    train(11,[broken,continued],[[0],[0,2]])
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'text.txt';p.write_text('凛音',encoding='utf-8')
        with p.open(encoding='utf-8') as f:content=f.read()
        normal_closed=f.closed
        try:
            with p.open(encoding='utf-8') as g:raise RuntimeError('intentional QA')
        except RuntimeError:pass
        train(12,[normal_closed,g.closed],[True,True],'上下文关闭资源；不消除块内异常。')
        train(14,[content,p.read_bytes().decode('utf-8')],['凛音','凛音'],'明确encoding与实际字节一致；跨环境默认差异参考Python open文档。')
        original=Path.cwd()
        try:
            os.chdir(folder)
            train(26,Path('text.txt').read_text(encoding='utf-8'),'凛音','普通相对文件路径从当前工作目录解释。')
        finally:os.chdir(original)
    train(13,[json.loads('true'),json.dumps(True)],[True,'true'])
    train(15,[10,20,30,40][1:3],[20,30])
    train(16,[list(x) for x in zip([1,2,3],['a','b'])],[[1,'a'],[2,'b']])
    train(17,[x*2 for x in [1,3]],[2,6])
    train(18,list(dict.fromkeys([3,1,3,2])),[3,1,2],'本检查验证替代写法；set不保证输入顺序依据Python集合规范，不凭一次输出顺序推断。')
    train(19,[7//2,7/2,-7//2],[3,3.5,-4])
    a=[1];b=[1];train(20,[a is b,a==b],[False,True])
    a=[1,2];r=a.append(3);train(21,[a,r],[[1,2,3],None])
    a={'x':1};r=a.update({'y':2});train(22,[a,r,{'x':1}|{'y':2}],[{'x':1,'y':2},None,{'x':1,'y':2}])
    s='ab';r=s.upper();train(23,[s,r],['ab','AB'])
    train(24,[issubclass(KeyboardInterrupt,Exception),issubclass(SystemExit,Exception),issubclass(KeyboardInterrupt,BaseException)],[False,False,True])
    key='AI_WAIFU_QA_DELIBERATELY_MISSING_M6'
    if key in os.environ:raise RuntimeError('QA缺失键意外已配置，停止而不打印环境。')
    train(25,[os.getenv(key),os.getenv(key,'fallback')],[None,'fallback'])
    a=torch.tensor([1.],requires_grad=True);b=a.detach();c=b.clone()
    with torch.no_grad():b.add_(1)
    train(27,[b.requires_grad,a.item(),c.item(),a.data_ptr()==b.data_ptr()],[False,2.,1.,True])
    checks['m6-train-technical-28']={'passed':True,'method':'官方文档核查，未伪造跨设备确定性试验',
        'claim':'相同随机种子不保证跨版本/平台/CPU-GPU复现一致',
        'reference':'https://docs.pytorch.org/docs/2.8/notes/randomness.html'}
    x=[1,2];y=x.append(4);train(29,[x,y],[[1,2,4],None])
    try:{}['name']
    except KeyError as e:train(30,e.args[0],'name')
    dev(1,[type({}).__name__,type(set()).__name__],['dict','set'])
    dev(2,sum([True,False,True]),2)
    d={'x':1};keys=d.keys();d['y']=2;dev(3,list(keys),['x','y'])
    try:{[1]:1}
    except TypeError as e:list_error=type(e).__name__
    try:{(1,[]):1}
    except TypeError as e:tuple_error=type(e).__name__
    dev(4,[list_error,tuple_error],['TypeError','TypeError'],'列表不可哈希；含列表的元组也不可哈希，不是所有不可变外壳都能作键。')
    dev(5,['' or 'fallback',0 and 7,3 and 7],['fallback',0,7])
    return {'purpose':'训练与开发审核事实核对；未加载模型或训练','python_checks':checks,'torch_version':torch.__version__,
            'references':['https://docs.python.org/3.12/library/stdtypes.html','https://docs.python.org/3.12/library/functions.html#open',
                'https://docs.python.org/3.12/library/exceptions.html','https://docs.python.org/3.12/library/os.html#os.getenv',
                'https://docs.python.org/3.12/library/json.html','https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html',
                'https://docs.pytorch.org/docs/2.8/notes/randomness.html']}


if __name__=='__main__':
    result=run()
    target=ROOT/'results/m6_preparation/reviews/expansion_fact_checks.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('35个技术问题的最小检查/官方依据已保存，无模型训练。')
