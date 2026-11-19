"""Static independent form validation; does not require FEniCS."""
import ast
from pathlib import Path

root=Path(__file__).parent
originals={
  'temporal':root/'original/nsbiot_Neumann_Robin_temporal_sweep.py',
  'spatial':root/'original/nsbiot_Neumann_Robin_spatial_sweep.py',
}

def get_form(p, name):
    tree=ast.parse(p.read_text(),filename=str(p))
    defs=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            if any(isinstance(x,ast.Name) and x.id==name for x in node.targets):
                defs.append(node.value)
    assert len(defs)==1,(p,name,len(defs))
    return ast.unparse(defs[0])

for study,old in originals.items():
    rr=root/f'nsbiot_Robin_Robin_{study}_sweep.py'
    for name in ('F_f3','F_pd3'):
        assert get_form(old,name)==get_form(rr,name),f'{study} {name} changed'
    a=get_form(old,'F_pe3')
    b=get_form(rr,'F_pe3')
    expected='''+ BETA * dot(ksi, n_b) * dot(zeta, n_b) * ds_b(5) - BETA * dot(2.0 * ksi_old - ksi_old1, n_b) * dot(zeta, n_b) * ds_b(5)'''
    assert expected in b, f'{study} missing mechanical terms'
    assert b.replace(' '+expected,'')==a, f'{study} has an unexpected mechanical change'
    print('PASS',study,'fluid+Darcy unchanged; exactly two normal mechanics terms added')

src=root/'nsbiot_forms.py'
comp=get_form(src,'F_pe3')
assert 'beta * dot(ksi, n_b)' in comp
assert 'beta * dot(2.0 * ksi_old - ksi_old1' in comp
print('PASS unified beta-parameter mechanics with beta=0 Neumann, beta>0 Robin')

# Check the unified forms term by term, independently of their grouping.
def signed_terms(node, sign=1):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return (signed_terms(node.left, sign) +
                signed_terms(node.right, -sign if isinstance(node.op, ast.Sub) else sign))
    return [(sign, ast.dump(node, include_attributes=False))]

from collections import Counter
for name in ('F_f3', 'F_pe3', 'F_pd3'):
    reference = get_form(originals['temporal'], name)
    unified = get_form(src, name)
    for source in ('f_f', 'f_e', 'f_p'):
        unified = unified.replace(source + '_source', source)
    if name == 'F_pe3':
        unified = unified.replace('beta', 'BETA')
        reference = get_form(root/'nsbiot_Robin_Robin_temporal_sweep.py', name)
    assert Counter(signed_terms(ast.parse(unified, mode='eval').body)) == Counter(
        signed_terms(ast.parse(reference, mode='eval').body)), name
    print('PASS unified', name, 'matches reference signed terms')
