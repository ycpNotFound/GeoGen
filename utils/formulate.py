import itertools
import random
import re

import sympy
from sympy import Eq, Float, Integer, nsimplify, simplify, solve, symbols, Integer
from formalgeo_v2.parse.basic import parse_equal_predicate
from .preset import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2, PREDICATES_REL_3, PREDICATES_ATTR,
                     SYMBOL_MAPPING_1, SYMBOL_MAPPING_2)
from .symbolic import parse_clause

def inverse_parse_one_theorem(theorem, parsed_theorem_GDL):
    """
    Inverse parse theorem to formal language.
    >> inverse_parse_one_theorem('t_name', 1, ('R', 'S', 'T'), parsed_theorem_GDL)
    't_name(1,RST)'
    >> inverse_parse_one_theorem('t_name', None, ('R', 'S', 'T'), parsed_theorem_GDL)
    't_name(RST)'
    >> inverse_parse_one_theorem('t_name', None, None, parsed_theorem_GDL)
    't_name'
    """
    t_name, t_branch, t_para = theorem
    if t_para is None:
        if t_branch is None:
            return t_name
        return "{}({})".format(t_name, t_branch)
    else:
        result = []
        i = 0
        for l in parsed_theorem_GDL[t_name]["para_len"]:
            result.append("")
            for _ in range(l):
                result[-1] += t_para[i]
                i += 1
        t_para = ",".join(result)

        if t_branch is None:
            return "{}({})".format(t_name, t_para)
        return "{}({},{})".format(t_name, t_branch, t_para)


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
            {s: symbols(str(s).upper()) for s in expr.free_symbols
             if '_' in str(s)}
        )
        expr = str(expr)
        expr = expr.replace('MA_', '\\angle ').replace('LL_', '')
        pattern = r'sqrt\((.*?)\)'
        expr = re.sub(pattern, r'\\sqrt{\1}', expr)
        return expr

    lhs_expr = process_expr(lhs_expr)
    rhs_expr = process_expr(rhs_expr)

    # test_expr = latex_to_sympy(f"{lhs_expr} = {rhs_expr}")
    natural = f"$ {lhs_expr} = {rhs_expr} $"
    return natural

def latex_to_sympy(latex_str, problem=None):
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
        if problem is not None:
            s = tuple(s.upper())
            sym = problem.get_sym_of_attr('MeasureOfAngle', s)
            return str(sym)
        else:
            return f'ma_{s}'
    
    # replace AB -> ll_ab
    def replace_two_letters(match):
        s = match.group(0).lower()
        if problem is not None:
            s = tuple(s.upper())
            sym = problem.get_sym_of_attr('LengthOfLine', s)
            return str(sym)
        else:
            return f'll_{s}'
    
    latex_str = re.sub(r'\\sqrt{([^}]*)}', replace_sqrt, latex_str)
    latex_str = re.sub(r'\\angle [A-Z]{3}', replace_three_letters, latex_str)
    latex_str = re.sub(r'[A-Z]{2}', replace_two_letters, latex_str)
    
    lhs, rhs = latex_str.split('=') if '=' in latex_str else (latex_str, '0')
    expr = simplify(f"({lhs})-({rhs})")
    return expr

def formulate_eqs_simple_temp(eq_str_dict, eq_str_list):
    
    long_eqs = [eq for eq in eq_str_list if '+' in eq or '-' in eq]
    short_eqs = [eq for eq in eq_str_list if eq not in long_eqs]
    if len(long_eqs) == 0:
        formulated_str = f"- <because> {', '.join(short_eqs)}, " 
    else:
        formulated_str = f"- Substitute {', '.join(short_eqs)} into equation:\n{long_eqs[0]}." 

    return formulated_str

def formulate_eqs_simple(eq_str_dict):
    eq_str_list = []
    for k, v in eq_str_dict.items():
        for eq in v:
            eq_str_list.append(eq)

    formulated_str = f"- <because>:" 
    for eq in eq_str_list:
        formulated_str += f"- \n{eq}."

    return formulated_str

# def formulate_eqs(eq_str_dict, target_str, problem):
#     eq_str_list = []
#     eq_str_premise_step = []
#     for k, v in eq_str_dict.items():
#         for eq in v:
#             eq_str_list.append(eq)
#             eq_str_premise_step.append(k)
#     long_eq_num = sum([1 if '+' in eq or '-' in eq else 0 for eq in eq_str_list])

#     if long_eq_num <= 1:
#         return formulate_eqs_simple(eq_str_dict, eq_str_list)
#     else:
#         return formulate_eqs_simple_v2(eq_str_dict, eq_str_list)
    

def find_minimal_equation_subset(eq_list, idx_list, target_symbol):
    if len(eq_list) in [1, 2]:
        return idx_list
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


def to_lower_with_spaces(s):
    result = ""
    for char in s:
        # 如果字符是大写且不是第一个字符，则在前面加空格
        if char.isupper() and result:
            result += " "
        result += char.lower()
    return result



def clause_to_nature_language(clauses, 
                              natural_template, 
                              upper=True, 
                              replace_sym=False,
                              replace_sym_mode=None):
    assert replace_sym_mode in [None, 'random', 'nature', 'math']
    conditions = []
    if replace_sym:
        if replace_sym_mode == 'random':
            symbol_mapping = random.choice(
                [SYMBOL_MAPPING_1, SYMBOL_MAPPING_2]
            )
        elif replace_sym_mode == 'nature':
            symbol_mapping = SYMBOL_MAPPING_1
        else:
            symbol_mapping = SYMBOL_MAPPING_2

    for clause in clauses:
        if 'Value' in clause:
            clause = clause.replace('Value', 'Equal')
        if 'Equal' in clause:
            condition_i = formulate_equal_clause(clause)
        else:
            try:
                pred, items = parse_clause(clause)
            except Exception as e:
                print('clause: ', clause)
                raise e
            
            if 'Shape' in clause:
                continue
            elif 'Circle' in clause:
                condition_i = f"\\odot {items[0]}"
            elif 'Collinear' in clause:
                points = items[0]
                condition_i = random.choice([
                    f"{', '.join(points)} lie on the same line",
                    f"{', '.join(points)} are collinear",
                    f"{', '.join(points)} are in one line",
                    f"{points[1]} lie on line {points[0]}{points[2]}",
                    f"{points[1]} is on the line {points[0]}{points[2]}"
                ])
            elif 'Cocircular' in clause:
                if len(items) == 1:
                    condition_i = f'circle {items[0]}'
                else:
                    circle, points = items
                    condition_i = random.choice([
                        f"{', '.join(points)} lie on the circle {circle}",
                        f"{', '.join(points)} lie on circle centered at {circle}",
                        f"{', '.join(points)} are on circle {circle}"
                    ])
            elif pred in PREDICATES_ENT:
                template = random.choice(natural_template[pred])
                condition_i = template.format(points=items[0])
            elif pred in PREDICATES_REL + PREDICATES_REL_2 + PREDICATES_REL + PREDICATES_REL_3:
                template = random.choice(natural_template[pred])
                condition_i = template.format(p1=items[0], p2=items[1])
            elif 'Mirror' in clause:
                pred = pred.replace('Mirror', '')
                template = random.choice(natural_template[pred])
                p1 = items[0]
                p2 = ''.join([list(items[1])[0]] + list(items[1])[1:][::-1])
                condition_i = template.format(p1=p1, p2=p2)
                
            elif 'Polygon' == pred:
                condition_i = f"{items[0]} is a polygon"
            elif 'Equation' == pred:
                condition_i = sympy_to_latex(items[0])
            elif 'Line' == pred:
                condition_i = f"line {items[0]}"
            elif 'Angle' == pred:
                condition_i = f"\\angle {items[0]}"
            elif 'Arc' == pred:
                condition_i = f"\\arc {items[0]}"
            
            else:
                raise KeyError(clause)
            if upper:
                condition_i = condition_i[0].upper() + condition_i[1:]
            if replace_sym and pred != 'Equation':
                for k, v in symbol_mapping.items():
                    condition_i = condition_i.replace(k, v)
            
        conditions.append(condition_i)
            
    return conditions



def formulate_equal_clause(clause):
    assert 'Equal' in clause
    eq_tree, attrs = parse_equal_predicate(clause)
    left_tree, right_tree = eq_tree[1][0], eq_tree[1][1]
    left_str = formulate_tree(left_tree)
    right_str = formulate_tree(right_tree)
    return f"{left_str} = {right_str}"


def formulate_tree(tree):
    if isinstance(tree, str):
        return tree
    if len(tree) == 1:
        return tree[0]
    name = tree[0]
    if name in ['Add', 'Mul', 'Sub', 'Div', 'Pow', 'Mod']:
        left_str = formulate_tree(tree[1][0])
        right_str = formulate_tree(tree[1][1])
        mid_symbol = {
            'Add': '+',
            'Mul': '*',
            'Sub': '-',
            'Div': '/',
            'Pow': '^',
            'Mod': '%'
        }[name]
        return f"{left_str} {mid_symbol} {right_str}"
    elif len(tree) == 2:
        name, items = tree[0], tree[1]
        if name == 'sqrt':
            tree_str = f"\\sqrt{{{items[0]}}}"
        elif name == 'MeasureOfAngle':
            tree_str = f"\\angle {''.join(items)}"
        elif name == 'LengthOfLine':
            tree_str = f"{''.join(items)}"
        elif name in ['PerimeterOfTriangle', 'PerimeterOfQuadrilateral',
                      'PerimeterOfCircle', 'PerimeterOfSector']:
            tree_str = f"C_{''.join(items)}"
        elif name in ['AreaOfTriangle', 'AreaOfQuadrilateral',
                      'AreaOfCircle', 'AreaOfSector']:
            tree_str = f"S_{''.join(items)}"
        elif name in ['HeightOfTriangle', 'HeightOfQuadrilateral']:
            tree_str = f"h_{''.join(items)}"
        elif name == 'RatioOfSimilarTriangle':
            tri_1, tri_2 = items[0][:3], items[0][3:]
            tree_str = f"k_{{{''.join(tri_1)}}},{{{''.join(tri_2)}}}"
        elif name == 'RatioOfSimilarQuadrilateral':
            quad_1, quad_2 = items[0][:4], items[0][4:]
            tree_str = f"k_{{{''.join(quad_1)}}}{{{''.join(quad_2)}}}"
        elif name == 'RatioOfSimilarArc':
            arc_1, arc_2 = items[0][:3], items[0][3:]
            tree_str = f"k_{{{''.join(arc_1[1:])}}},{{{''.join(arc_2[1:])}}}"
        elif name == 'RatioOfMirrorSimilarTriangle':
            tri_1, tri_2 = items[0][:3], items[0][3:]
            tree_str = f"k_{{{''.join(tri_1)}}},{{{''.join(tri_2[::-1])}}}"
        elif name == 'RatioOfMirrorSimilarQuadrilateral':
            quad_1, quad_2 = items[0][:4], items[0][4:]
            tree_str = f"k_{{{''.join(quad_1)}}},{{{''.join(quad_2[::-1])}}}"
        elif name == 'LengthOfArc':
            tree_str = f"⌒{''.join(items[1:])}"
        elif name == 'MeasureOfArc':
            tree_str = f"m⌒{''.join(items[1:])}"
        elif name == 'RadiusOfCircle':
            tree_str = f"r_{''.join(items)}"
        elif name == 'DiameterOfCircle':
            tree_str = f"d_{''.join(items)}"

        else:
            raise KeyError(name)
    else:
        raise KeyError(len(tree))
    
    return tree_str


def clause_to_nature_language_ori(clauses, 
                              natural_template, 
                              upper=True, 
                              replace_sym=False,
                              replace_sym_mode=None):
    assert replace_sym_mode in [None, 'random', 'nature', 'math']
    conditions = []
    if replace_sym:
        if replace_sym_mode == 'random':
            symbol_mapping = random.choice(
                [SYMBOL_MAPPING_1, SYMBOL_MAPPING_2]
            )
        elif replace_sym_mode == 'nature':
            symbol_mapping = SYMBOL_MAPPING_1
        else:
            symbol_mapping = SYMBOL_MAPPING_2

    for clause in clauses:
        try:
            pred, items = parse_clause(clause)
        except Exception as e:
            print('clause: ', clause)
            raise e
        if 'Value' in clause:
            clause = clause.replace('Value', 'Equal')
        if 'Equal' in clause:
            if 'sqrt' in items[1]:
                items = tuple([
                    items[0],
                    items[1].replace('sqrt', '\\sqrt')
                ])
                
            if pred == 'MeasureOfAngle':
                if items[1].isalpha() and len(items[1]) > 1:
                    condition_i = f"\\angle {items[0]} = \\angle {items[1]}"
                else:
                    if '+' in items[1] or '-' in items[1]:
                        condition_i = f"\\angle {items[0]} = ({items[1]})°"
                    else:
                        condition_i = f"\\angle {items[0]} = {items[1]}°"
            elif pred == 'LengthOfLine':
                condition_i = f"{items[0]} = {items[1]}"
            elif pred == 'LengthOfArc':
                condition_i = f"\\arc {items[0]} = \\arc {items[1]}"
            elif pred == 'MeasureOfArc':
                # OAB -> BOA
                angle_1 = ''.join([items[0][2], items[0][0], items[0][1]])
                if len(items[1]) == 3:
                    angle_2 = ''.join([items[1][2], items[1][0], items[1][1]])
                    condition_i = f"\\angle {angle_1} = \\angle {angle_2}"
                else:
                    if '+' in items[1] or '-' in items[1]:
                        condition_i = f"\\angle {angle_1} = ({items[1]})°"
                    else:
                        condition_i = f"\\angle {angle_1} = {items[1]}°"
            elif pred == 'RadiusOfCircle':
                condition_i = f'radius of \\odot {items[0]} = {items[1]}'
            elif pred == 'AreaOfTriangle':
                condition_i = f"area of \\triangle {items[0]} = {items[1]}"
            elif pred == 'DiameterOfCircle':
                condition_i = f"diameter of \\odot {items[0]} = {items[1]}"
            elif pred == "AreaOfQuadrilateral":
                condition_i = f"area of {items[0]} = {items[1]}"
            elif pred == 'PerimeterOfTriangle':
                condition_i = f"perimeter of \\triangle {items[0]} = {items[1]}"
            elif pred == "PerimeterOfQuadrilateral":
                condition_i = f"perimeter of {items[0]} = {items[1]}"
            elif pred == 'RatioOfSimilarTriangle':
                tri_1, tri_2 = items[0][:3], items[0][3:]
                condition_i = f"ratio of similar \\triangle {tri_1} and \\triangle {tri_2} = {items[1]}"
            elif pred == 'RatioOfMirrorSimilarTriangle':
                tri_1, tri_2 = items[0][:3], items[0][3:]
                tri_2 = ''.join([list(tri_2)[0]] + list(tri_2)[1:][::-1])
                condition_i = f"ratio of similar \\triangle {tri_1} and \\triangle {tri_2} = {items[1]}"
            elif pred == 'RatioOfSimilarQuadrilateral':
                quad_1, quad_2 = items[0][:4], items[0][4:]
                condition_i = f"ratio of similar quadrilateral {quad_1} and {quad_2} = {items[1]}"
            
            elif pred == 'HeightOfQuadrilateral':
                condition_i = f"height of {items[0]} = {items[1]}"
            elif pred == 'AreaOfCircle':
                condition_i = f"area of \\odot {items[0]} = {items[1]}"
            elif pred == 'RatioOfSimilarArc':
                arc_1, arc_2 = items[0][:3], items[0][3:]
                condition_i = f"ratio of \\arc {arc_1} and {arc_2} = {items[1]}"
            elif pred == "AreaOfSector":
                condition_i = f"area of sector {items[0]} = {items[1]}"
            elif pred == 'Equal': # fail to parse
                condition_dict = {
                    "LengthOfLine": "{item}",
                    "MeasureOfAngle": "\\angle {item}",
                    "RadiusOfCircle": "radius of \\odot {item}",
                    "MeasureOfArc": "\\arc {item}"
                }
                
                clause_l, clause_r = items[0], items[1] 
                
                if any(pred in clause_r for pred in PREDICATES_ATTR):
                    # Equal, MeasureOfAngle(..), MeasureOfArc(..)
                    pred_l, item_l = parse_clause(clause_l)
                    if pred_l in condition_dict:
                        condition_l = condition_dict[pred_l].format(item=item_l[0])
                    else:
                        condition_l = to_lower_with_spaces(pred_l) + f" {item_l[0]}"
                else:
                    condition_l = clause_l
                
                if any(pred in clause_r for pred in PREDICATES_ATTR):
                    pred_r, item_r = parse_clause(clause_r)
                    if pred_r in condition_dict:
                        condition_r = condition_dict[pred_r].format(item=item_r[0])
                    else:
                        condition_r = to_lower_with_spaces(pred_r) + f" {item_r[0]}"
                else:
                    condition_r = clause_r
                
                condition_i = f"{condition_l} = {condition_r}"
            else:
                raise KeyError(pred)
        elif 'Shape' in clause:
            continue
        elif 'Circle' in clause:
            condition_i = f"\\odot {items[0]}"
        elif 'Collinear' in clause:
            points = items[0]
            condition_i = random.choice([
                f"{', '.join(points)} lie on the same line",
                f"{', '.join(points)} are collinear",
                f"{', '.join(points)} are aligned in one line",
                f"{points[1]} lie on line segment {points[0]}{points[2]}",
                f"{points[1]} is on the line segment {points[0]}{points[2]}"
            ])
        elif 'Cocircular' in clause:
            if len(items) == 1:
                condition_i = f'circle {items[0]}'
            else:
                circle, points = items
                condition_i = random.choice([
                    f"{', '.join(points)} lie on the circle {circle}",
                    f"{', '.join(points)} lie on the same circle centered at {circle}",
                    f"{', '.join(points)} are on circle {circle}"
                ])
        elif pred in PREDICATES_ENT:
            template = random.choice(natural_template[pred])
            condition_i = template.format(points=items[0])
        elif pred in PREDICATES_REL + PREDICATES_REL_2 + PREDICATES_REL + PREDICATES_REL_3:
            template = random.choice(natural_template[pred])
            condition_i = template.format(p1=items[0], p2=items[1])
        elif 'Mirror' in clause:
            pred = pred.replace('Mirror', '')
            template = random.choice(natural_template[pred])
            p1 = items[0]
            p2 = ''.join([list(items[1])[0]] + list(items[1])[1:][::-1])
            condition_i = template.format(p1=p1, p2=p2)
            
        elif 'Polygon' == pred:
            condition_i = f"{items[0]} is a polygon"
        elif 'Equation' == pred:
            condition_i = sympy_to_latex(items[0])
        elif 'Line' == pred:
            condition_i = f"line {items[0]}"
        elif 'Angle' == pred:
            condition_i = f"\\angle {items[0]}"
        elif 'Arc' == pred:
            condition_i = f"\\arc {items[0]}"
        
        else:
            raise KeyError(clause)
        if upper:
            condition_i = condition_i[0].upper() + condition_i[1:]
        if replace_sym and pred != 'Equation':
            for k, v in symbol_mapping.items():
                condition_i = condition_i.replace(k, v)
        conditions.append(condition_i)
        
    return conditions

def convert_upper_to_lower(name): 
    # IsoscelesRightTriangle -> isosceles right triangle
    converted_name = name[0].lower()
    for char in name[1:]:
        if char.isupper():
            converted_name += ' '
            char = char.lower()
        converted_name += char
    return converted_name


