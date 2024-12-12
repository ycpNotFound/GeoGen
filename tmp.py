from sympy import simplify, nsimplify, Integer, Float, symbols, linear_eq_to_matrix, solve, Eq
import re
import numpy as np
from formalgeo.core.engine import EquationKiller as EqKiller
import requests
import json

def sympy_to_latex(expr):
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
    expr = simplify(f"{lhs}-{rhs}")
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
    eq_list = []
    for eq in eq_str_list + [target_str]:
        eq = eq.replace('∠', '\\angle').replace('°', '')
        if '$' in eq:
            eq = re.findall(r'\$(.*?)\$', eq)[0]
        elif 'from' in eq:
            eq = eq.split('from')[0]
        eq = latex_to_sympy(eq)
        eq_list.append(eq)
    
    target_sym = eq_list[-1].free_symbols.pop()
    eq_list = eq_list[:-1]
    
    # substitute value in eq_list
    sym2value = {}
    for eq in eq_list:
        if len(eq.free_symbols) == 1:
            sym = eq.free_symbols.pop()
            res = solve(Eq(eq, 0), sym, dict=True)[0]
            sym2value[sym] = res[sym]
    print(sym2value)

    eq_list = [
        eq.subs(sym2value) for eq in eq_list
        if len(eq.subs(sym2value).free_symbols) > 0
    ]
    # build sym -> eqs map
    sym2eqs = {}
    for eq in eq_list:
        for sym in list(eq.free_symbols):
            if sym not in sym2eqs:
                sym2eqs[sym] = [eq]
            else:
                sym2eqs[sym].append(eq)
    sym2eqs_char = {}
    sym2char = {original: symbols(chr(97 + idx)) for idx, original in enumerate(sym2eqs)}
    for k, v in sym2eqs.items():
        sym2eqs_char[sym2char[k]] = [expr.subs(sym2char) for expr in v]
    eqs_char = [expr.subs(sym2char) for expr in eq_list]
    print(sym2eqs)
    print(sym2eqs_char)
    
    res_1 = EqKiller.simplification_sym_replace(eqs_char, sym2char[target_sym])
    EqKiller.get_minimum_target_equations(sym2char[target_sym], eqs_char)
    print(eqs_char)
    return
    
    
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

if __name__ == '__main__':
    eq_str_list = ['∠ EFH = ∠ HFG from step 5', '$ ∠ EFG = 2*∠ EFH $ from step 5', '$ ∠ EFG = 2*∠ HFG $ from step 5', '∠ FGH = ∠ HGE from step 5', '$ ∠ FGE = 2*∠ FGH $ from step 5', '$ ∠ FGE = 2*∠ HGE $ from step 5', '∠ GEF = 30°', '∠ GEH = 15° from step 7', '∠ HEF = 15° from step 6', '$ ∠ EFG + ∠ FGE + ∠ GEF = 180 $ from step 4', '$ ∠ EFH + ∠ FHE + ∠ HEF = 180 $ from step 3', '$ ∠ EHG + ∠ GEH + ∠ HGE = 180 $ from step 2', '$ ∠ FGH + ∠ GHF + ∠ HFG = 180 $ from step 1']
    target_str = '∠ GHF = 105°'
    # formulate_eqs(eq_str_list, target_str)
    # test_wolframe_alpha()
    test_sympy_subs()
