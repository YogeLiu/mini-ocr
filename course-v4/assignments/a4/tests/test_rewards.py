import pytest
from ._impl import adapters as A

# 受控工具：字符替换误差可手算，解析/格式判定不依赖外部渲染版本。
TOOLS = dict(parse=lambda s:s != 'INVALID', normalize=lambda s:''.join(s.split()),
             error_rate=lambda target,s:sum(a!=b for a,b in zip(target,s))/len(target)+abs(len(target)-len(s))/len(target),
             format_issues=lambda s:['prose'] if s.startswith('Here') else [])


@pytest.mark.parametrize('completion,want',[('',0),('INVALID',0),('Here abc',.1),('abcd',1),('abxd',.3),('abxy',.1),('zzzz',.1),('a b c d',1)])
def test_reward_branches(completion,want):
    assert A.reward(completion,'abxy','abcd',**TOOLS) == pytest.approx(want)


def test_clean_source_and_parse_only_counterexample():
    assert A.reward('abxd','abcd','abcd',**TOOLS) == .1
    assert A.reward('x','abxy','abcd',mode='parse_only',**TOOLS) == 1
    assert A.reward('x','abxy','abcd',**TOOLS) == .1
    assert A.reward('INVALID','abxy','abcd',mode='parse_only',**TOOLS) == 0
    assert A.reward('INVALID','INVALID','INVALID',**TOOLS) == 0  # 验证器盲区：标准答案也不保证满分


def test_probe_contract_and_no_universal_safety_claim():
    assert A.hack_probes('abxy','abcd') == [('empty',''),('single_symbol','x'),('copy_source','abxy'),
        ('copy_with_prose','Here is the corrected formula: abxy'),('repeat_source','abxy abxy abxy'),('gold_answer','abcd')]
    assert A.reward('x','z','x',**TOOLS) == 1  # 单字符也可能恰好正确，不能一概判投机


@pytest.mark.parametrize('which',range(4))
def test_reward_errors(which):
    calls=[lambda:A.reward(None,'x','y',**TOOLS),lambda:A.reward('x','x','y',mode='bad',**TOOLS),
           lambda:A.hack_probes(1,'y'),lambda:A.reward('z','x','y',**{**TOOLS,'error_rate':lambda a,b:float('nan')})]
    with pytest.raises(ValueError): calls[which]()
