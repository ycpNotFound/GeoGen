from sympy import simplify, nsimplify, Integer, Float, symbols, linear_eq_to_matrix, solve, Eq, Matrix
import re
import numpy as np
from formalgeo.core.engine import EquationKiller as EqKiller
import requests
import json
import itertools

def sympy_to_latex(expr):
    # equation: expr = 0
    lhs, rhs = expr.split('=') if '=' in expr else (expr, '0')
    expr = simplify(lhs + '-(' + rhs + ')')

    lhs_terms, rhs_terms = [], []
    terms = expr.as_ordered_terms()
    for term in terms:
        if term.as_coeff_Mul()[0] > 0:
            lhs_terms.append(term)
        else: 
            rhs_terms.append(-term)

    lhs_expr = nsimplify(sum(lhs_terms))
    rhs_expr = nsimplify(sum(rhs_terms))

    def process_expr(expr):
        if isinstance(expr, (Integer, Float)):
            return expr
        expr = expr.xreplace(
            {s: symbols(str(s).upper()) for s in expr.free_symbols}
        )
        expr = str(expr)
        expr = expr.replace('MA_', '\\angle ').replace('LL_', '')
        expr = expr.replace('*', '')
        pattern = r'sqrt\((.*?)\)'
        expr = re.sub(pattern, r'\\sqrt{\1}', expr)
        return expr

    lhs_expr = process_expr(lhs_expr)
    rhs_expr = process_expr(rhs_expr)

    # test_expr = latex_to_sympy(f"{lhs_expr} = {rhs_expr}")
    natural = f"$ {lhs_expr} = {rhs_expr} $"
    return natural

def latex_to_sympy(latex_str):
    # replace \\sqrt{} -> sqrt()
    def replace_sqrt(match):
        inside = match.group(1)
        # 只把字母部分转小写
        inside_lower = re.sub(r'[A-Za-z]+', lambda m: m.group(0).lower(), inside)
        return f'sqrt({inside_lower})'
    
    # replace \\angle ABC -> ma_abc
    def replace_three_letters(match):
        s = match.group(0).lower()
        s = s.split(' ')[-1]
        return f'ma_{s}'
    
    # replace AB -> ll_ab
    def replace_two_letters(match):
        s = match.group(0).lower()
        return f'll_{s}'
    
    latex_str = re.sub(r'\\sqrt{([^}]*)}', replace_sqrt, latex_str)
    latex_str = re.sub(r'\\angle [A-Z]{3}', replace_three_letters, latex_str)
    latex_str = re.sub(r'[A-Z]{2}', replace_two_letters, latex_str)
    
    lhs, rhs = latex_str.split('=') if '=' in latex_str else (latex_str, '0')
    expr = simplify(f"({lhs})-({rhs})")
    return expr

def simplification_value_replace(eqs):
    """
    Simplify equations by replacing sym with known value.
    :param problem: Instance of class <Problem>.
    """
    sym2value = {}

    update = True
    while update:
        update = False
        remove_lists = set()  # equation to be deleted
        add_lists = []  # equation to be added

        for eq in eqs:  # solve eq that only one sym unsolved
            if len(eq.free_symbols) != 1:
                continue

            target_sym = list(eq.free_symbols)[0]

            result = solve(Eq(eq, 0), target_sym, dict=True)[0]  # solve equations

            if target_sym in result:
                sym2value[target_sym] = result[target_sym]
                remove_lists |= {eq}

        for eq in eqs:  # value replace
            if eq in remove_lists:
                continue
            # if len(eq.free_symbols) != 1:
            #     continue
            raw_eq = eq
            simplified = False
            added_premise = []
            for sym in eq.free_symbols:
                if sym not in sym2value or sym2value[sym] is None:
                    continue
                simplified = True  # replace sym with value when the value known
                eq = eq.subs(sym, sym2value[sym])

                remove_lists |= {raw_eq}

            if not simplified:
                continue

            if len(eq.free_symbols) == 0:  # no need to add new simplified equation when it's all sym known
                continue
            else:  # add new simplified equation
                update = True

                add_lists.append((eq, 0))

        for remove_eq in remove_lists:  # remove useless equation
            eqs.remove(remove_eq)
        for add_eq, premise in add_lists:  # remove useless equation
            eqs.append(add_eq)
            
    return eqs

def formulate_eqs(eq_str_list, target_str):
    expr_list = []
    for eq in eq_str_list + [target_str]:
        eq = eq.replace('∠', '\\angle').replace('°', '')
        if '$' in eq:
            eq = re.findall(r'\$(.*?)\$', eq)[0]
        elif 'from' in eq:
            eq = eq.split('from')[0]
        expr = latex_to_sympy(eq)
        expr_list.append(expr)
    
    target_sym = expr_list[-1].free_symbols.pop()
    expr_list = expr_list[:-1]
    
    # substitute value in eq_list
    sym_value_premise_list = []
    for idx, expr in enumerate(expr_list):
        if len(expr.free_symbols) == 1:
            sym = expr.free_symbols.pop()
            res = solve(Eq(expr, 0), sym, dict=True)[0]
            sym_value_premise_list.append([sym, res[sym], idx])

    expr_subs_list, expr_subs_ids = [], []
    expr_id2premise = {}
    for i, expr in enumerate(expr_list):
        if i in [item[2] for item in sym_value_premise_list]:
            continue
        premise_ids = []

        for sym, value, j in sym_value_premise_list:
            if sym in expr.free_symbols:
                expr = expr.subs({sym: value})
                premise_ids.append(j)

        if len(expr.free_symbols) > 0:
            expr_subs_list.append(expr)
            expr_subs_ids.append(i)
            expr_id2premise[i] = premise_ids
    
    # find miniset of eqs
    eqs = [ Eq(expr, 0) for expr in expr_subs_list]
    subset_ids = find_minimal_equation_subset(eqs, expr_subs_ids, target_sym)

    if len(subset_ids) == 0 or len(subset_ids) > 3:
        return None

    formulated_str = ''
    for idx_ori in subset_ids:
        idx_subs = expr_subs_ids.index(idx_ori)
        premise_ids = expr_id2premise[idx_ori]
        if len(premise_ids) > 0:
            # has changed by substitute value
            premise_str = ', '.join([eq_str_list[i] for i in premise_ids])
            extend_eq = eqs[idx_subs]
            extend_str = sympy_to_latex(str(extend_eq.lhs))
            formulated_str += f"- Substitute {premise_str} in {eq_str_list[idx_ori]}, <therefore> {extend_str}.\n"

        else:
            extend_str = eq_str_list[idx_ori]
            if 'from' in extend_str:
                items = [s.strip() for s in extend_str.split('from')]
                statement, step = items[0], items[-1]
                extend_str = f"From {step}: {statement}"
            formulated_str += f"- {extend_str}.\n"
        
    formulated_str += '- Solving equation groups: \n'
    for idx_ori in subset_ids:
        idx_subs = expr_subs_ids.index(idx_ori)
        premise_ids = expr_id2premise[idx_ori]
        extend_eq = eqs[idx_subs]
        extend_str = sympy_to_latex(str(extend_eq.lhs))
        formulated_str += f'\t- {extend_str}.\n'


    return formulated_str

def find_minimal_equation_subset(eq_list, idx_list, target_symbol):
    for subset_size in range(1, len(eq_list) + 1):
        for subset in itertools.combinations(eq_list, subset_size):
            # Try solving the target_symbol using the subset
            solution = solve(subset)
            if target_symbol in solution:
                target_value = solution.get(target_symbol)
                if isinstance(target_value, (int, float, Float, Integer)):
                    idx_subset = [idx_list[eq_list.index(eq)] for eq in subset]
                    return idx_subset
    return []
    
def test_wolframe_alpha():
    app_id = "GUL4RH-486H6ERPR3"
    query = "solve x + 2y + z = 6, 2x - y + 3z = 14, y + 2z = 8 for x, y, z, step by step"
    url = f"http://api.wolframalpha.com/v1/query"
    params = {
        "appid": app_id,
        "input": query,
        "podstate": "Solution__Step-by-step solution",
        "output": "JSON",
        "stepbystep": True,
        "format": "plaintext"
    }


    response = requests.get(url, params=params).json()


    pods = response["queryresult"]["pods"]
    for pod in pods:
        print(pod["title"])
        for subpod in pod["subpods"]:
            print(subpod["plaintext"])
    print(json.dumps(pods[1], indent=4, ensure_ascii=False))

def test_sympy_subs():
    from sympy import symbols, UnevaluatedExpr
    x, a, b = symbols('x a b')
    def substitute_without_simplification(expr, substitutions):
        for old, new in substitutions.items():
            expr = expr.subs(old, UnevaluatedExpr(new))
        return expr
    # 示例表达式
    expr = a - 1/2 * b
    # 替换规则
    substitutions = {a: 2*x-1, b: x+1}
    # 执行替换但不简化
    new_expr = substitute_without_simplification(expr, substitutions)
    new_expr_symp = expr.subs({a:(2*x-1), b: (x+1)})

    print(new_expr)
    
def test_parse():
    from utils.symbolic import parse_clause
    
    clause = "Equal(LengthOfLine(EB),RadiusOfCircle(E))"
    pred_name, items = parse_clause(clause)
    
    assert pred_name == "Equal"
    pred_l, item_l = parse_clause(items[0])
    pred_r, item_r = parse_clause(items[1])
    condition_dict = {
        "LengthOfLine": "{item}",
        "MeasureOfAngle": "\\angle {item}",
        "RadiusOfCircle": "radius of \\circle {item}",
        "MeasureOfArc": "\\arc {item}"
    }
    condition_l = condition_dict[pred_l].format(item=item_l[0])
    condition_r = condition_dict[pred_r].format(item=item_r[0])
    condition_i = f"{condition_l} = {condition_r}"

if __name__ == '__main__':

    eq_str_list = [
        '∠ ABC = 90°', 
        '∠ BCD = 90°', 
        '∠ CDA = 90°', 
        '∠ DAB = 90° from step 3', 
        '$ ∠ ABC + ∠ BCA + ∠ CAB = 180 $', 
        '$ ∠ BCD = ∠ ACD + ∠ BCA $ from given condition', 
        '$ ∠ ACD + ∠ CDA + ∠ DAC = 180 $ from step 2', 
        '$ ∠ DAB = ∠ CAB + ∠ DAC $ from step 1', 
        '∠ BCA = ∠ CAB from step 4'
    ]
    target_str = '∠ BCA = 45°'
    # formulate_eqs(eq_str_list, target_str)
    # test_wolframe_alpha()
    # test_sympy_subs()
    # filter_no_sqrt()
    test_parse()
