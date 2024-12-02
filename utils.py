import json
import math
import random
import re
from collections import Counter

import numpy as np
import sympy
from matplotlib import pyplot as plt
from sympy import Add, Eq, Float, Integer, Mul, Symbol, simplify, symbols, nsimplify

PREDICATES_PRE = [
    # Preset
    "Collinear",
    "Cocircular",
]
PREDICATES_ENT = [
    # Entity
    # "Circle",
    "Triangle",
    "Parallelogram",
    "Rectangle",
    "Rhombus",
    "RightTriangle",
    "Square",
    "EquilateralTriangle",
    "IsoscelesTriangle",
    "Trapezoid",
    "Kite",
    "RightTrapezoid",
    "IsoscelesTrapezoid",
    "IsoscelesRightTriangle",
]
PREDICATES_REL = [
    # Relation
    # "IsCentreOfCircle", 
    "PerpendicularBetweenLine",
    "ParallelBetweenLine",
    "IsTangentOfCircle",
    "IsDiameterOfCircle",
    "IsMidpointOfLine",
    "IsBisectorOfAngle",
    "IsPerpendicularBisectorOfLine",
    # "SimilarBetweenTriangle",
    # "SimilarBetweenQuadrilateral",
    "IsAltitudeOfTriangle",
    # "MirrorCongruentBetweenTriangle",
    # "CongruentBetweenTriangle",
    "IsMedianOfTriangle",
    "IsIncenterOfTriangle",
    "IsMidsegmentOfTriangle",
    # "MirrorSimilarBetweenTriangle",
    "IsMidpointOfArc",
    # "CongruentBetweenArc",
    "IsMidsegmentOfQuadrilateral",
    "IsCentroidOfTriangle",
    # "MirrorCongruentBetweenQuadrilateral",
    # "IsAltitudeOfQuadrilateral",
    # "CongruentBetweenQuadrilateral",
    "IsCircumcenterOfTriangle",
    # "IsIncenterOfQuadrilateral",
    "IsCircumcenterOfQuadrilateral",
]
PREDICATES_REL_2 = [
    "SimilarBetweenTriangle",
    # "MirrorSimilarBetweenTriangle",
    "SimilarBetweenQuadrilateral",
    "CongruentBetweenTriangle",
    # "MirrorCongruentBetweenTriangle",
    # "CongruentBetweenArc",
    "CongruentBetweenQuadrilateral",
    # "MirrorCongruentBetweenQuadrilateral",
    
]
PREDICATES_ATTR = [
    # Attribution
    "LengthOfLine",
    "MeasureOfAngle",
    "MeasureOfArc",
    "RadiusOfCircle",
    "AreaOfTriangle",
    "DiameterOfCircle",
    "AreaOfQuadrilateral",
    "PerimeterOfTriangle",
    "LengthOfArc",
    "PerimeterOfQuadrilateral",
    "RatioOfSimilarTriangle",
    "HeightOfQuadrilateral",
    "AreaOfSector",
    "AreaOfCircle",
    "RatioOfSimilarArc",
    "RatioOfMirrorSimilarTriangle",
    "RatioOfSimilarQuadrilateral",
]
PREDICATES = PREDICATES_PRE + PREDICATES_ENT + PREDICATES_ATTR + PREDICATES_REL + PREDICATES_REL_2

PREDICATES_TO_NAMES = {
    "PerpendicularBetweenLine": "perpendicular lines",
    "ParallelBetweenLine": "parallel lines",
    "IsTangentOfCircle": "tangent line",
    "IsDiameterOfCircle": "diameter of circle",
    "IsMidpointOfLine": "midpoint",
    "IsBisectorOfAngle": "angle bisector",
    "IsPerpendicularBisectorOfLine": "perpendicular bisector",
    "IsAltitudeOfTriangle": "altitude line",
    "IsMedianOfTriangle": "median line",
    "IsIncenterOfTriangle": "incenter",
    "IsMidsegmentOfTriangle": "midegment",
    "IsMidpointOfArc": "midpoint",
    "IsMidsegmentOfQuadrilateral": "midsegment",
    "IsCentroidOfTriangle": "centroid",
    "IsCircumcenterOfTriangle": "circumcenter",
    "IsCircumcenterOfQuadrilateral": "circumcenter",
    "SimilarBetweenTriangle": "similar triangles",
    "SimilarBetweenQuadrilateral": "similar quadrilaterals",
    "CongruentBetweenTriangle": "congruent triangles",
    "CongruentBetweenQuadrilateral": "congruent quadrilaterals"
}
PRESET_COLOR_PROBS = [0.5, 0.06, 0.06, 0.06, 0.06, 0.06, 0.2]
PRESET_COLORS = [
    # l: line | p: point | c: char | a: annoation | f: fill_in
    # Color: BGR
    {
        "l_color": "#000000",
        "p_color": "#2f5597",
        "c_color": "#0648cc",
        "a_color": "#f05015",
        "f_color": "#ffffff"   
    },
    {
        "l_color": "#2a2a2a",
        "p_color": "#000000",
        "c_color": "#2a2a2a",
        "a_color": "#2f5597",
        "f_color": "#D1D3D0"
        
    },
    {
        "l_color": "#005896",
        "p_color": "#00406C",
        "c_color": "#2a2a2a",
        "a_color": "#f05015",
        "f_color": "#9DC7DD"
    },
    {
        "l_color": "#638E50",
        "p_color": "#48683A",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#B5D69E"
    },
    {
        "l_color": "#D29A42",
        "p_color": "#956A23",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#FFE6B2"
    },
    {
        "l_color": "#6B4582",
        "p_color": "#442C52",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#C8B0C9"
    },
    {
        "l_color": "#282828",
        "p_color": "#000000",
        "c_color": "#282828",
        "a_color": "#282828",
        "f_color": "#ffffff"
    },
]

def hex_to_bgr(hex_str):
    rgb_color = tuple(int(hex_str[i:i+2], 16) for i in (1, 3, 5)) 
    bgr_color = rgb_color[::-1]  # Convert to BGR format 
    return bgr_color

def bgr_to_hex(b, g, r):
    # 将每个分量转换为两位的十六进制字符串
    rb = format(r, '02x')
    gb = format(g, '02x')
    bb = format(b, '02x')
    
    # 拼接成完整的十六进制颜色码
    hex_color = f"#{rb}{gb}{bb}"
    return hex_color


SYMBOL_MAPPING_1 = {
    "\\triangle": "triangle",
    "\\perp": "is perpendicular to",
    "\\parallel": "is parallel to",
    "\\odot": "circle",
    "\\angle": "angle",
    "\\arc": "arc",
    "\\sqrt": "sqrt of",
}
SYMBOL_MAPPING_2 = {
    "\\triangle": "△",
    "\\perp": "⊥",
    "\\parallel": "∥",
    "\\odot": "⊙",
    "\\angle": "∠",
    "\\arc": "⌒",
    "\\sqrt": "√"
}


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

def clause_to_nature_language(clauses, natural_template, upper=True, symbol2nature=None):
    conditions = []

    for clause in clauses:
        pred, items = parse_clause(clause)
        if 'Value' in clause:
            clause = clause.replace('Value', 'Equal')
        if 'Equal' in clause:
            if 'sqrt' in items[1]:
                items = tuple([
                    items[0],
                    items[1].replace('sqrt', '\\sqrt')
                ])
            if pred == 'MeasureOfAngle':
                if items[1].isalpha():
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
                if items[1].isalpha():
                    angle_2 = ''.join([items[1][2], items[1][0], items[1][1]])
                    condition_i = f"\\angle {angle_1} = \\angle {angle_2}"
                else:
                    if '+' in items[1] or '-' in items[1]:
                        condition_i = f"\\angle {angle_1} = ({items[1]})°"
                    else:
                        condition_i = f"\\angle {angle_1} = {items[1]}°"
            elif pred == 'RatioOfSimilarTriangle':
                tri_1, tri_2 = items[0][:3], items[0][3:]
                condition_i = f"ratio of similar \\triangle {tri_1} and \\triangle {tri_2} = {items[1]}"
            else:
                raise KeyError(pred)
        elif 'Shape' in clause:
            continue
        elif 'Collinear' in clause:
            points = items[0]
            condition_i = random.choice([
                f"points {', '.join(points)} lie on the same line",
                f"points {', '.join(points)} are collinear",
                f"points {', '.join(points)} are aligned in a straight line",
                f"the point {points[1]} lies on line segment {points[0]}{points[2]}",
                f"the point {points[1]} is on the line segment {points[0]}{points[2]}"
            ])
        elif 'Cocircular' in clause:
            circle, points = items
            condition_i = random.choice([
                f"points {', '.join(points)} lie on the circle {circle}",
                f"points {', '.join(points)} lie on the same circle centered at point {circle}",
                f"the points {', '.join(points)} are on circle {circle}"
            ])
        elif pred in PREDICATES_ENT:
            template = random.choice(natural_template[pred])
            condition_i = template.format(points=items[0])
        elif pred in PREDICATES_REL + PREDICATES_REL_2:
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
        conditions.append(condition_i)
    
    if symbol2nature is None:
        symbol_mapping = random.choice([SYMBOL_MAPPING_1, 
                                    SYMBOL_MAPPING_2])
    elif symbol2nature == True:
        symbol_mapping = SYMBOL_MAPPING_1
    else:
        symbol_mapping = SYMBOL_MAPPING_2
        
    conditions_res = []
    for c in conditions:
        for k, v in symbol_mapping.items():
            c = c.replace(k, v)
        conditions_res.append(c)
        
    return conditions_res

def convert_upper_to_lower(name): 
    # IsoscelesRightTriangle -> isosceles right triangle
    converted_name = name[0].lower()
    for char in name[1:]:
        if char.isupper():
            converted_name += ' '
            char = char.lower()
        converted_name += char
    return converted_name

def extract_sqrt_terms(expression):
    # 使用 SymPy 的 args 属性，该属性对于加法表达式返回所有加项，
    # 对于乘法表达式返回所有因子。
    if expression.func is Add:  # Add 是加法类
        res_list = []
        for arg in expression.args:
            res = extract_sqrt_terms(arg)
            if len(res) != 0:
                res_list += res
        return res_list

    elif expression.func is Mul:  # Mul 是乘法类
        res_list = []
        for arg in expression.args:
            res = extract_sqrt_terms(arg)
            if len(res) != 0:
                res_list += res
        return res_list

    elif expression.is_Pow and expression.exp == 1/2:
        return [expression]

    else:
        return []

def get_predicate_name(clause):
    if 'Equal' in clause:
        return 'Equal'
    else:
        return clause.split('(')[0]
    
def move_subtractions_to_rhs(eq):
    # 确保所有项都在等号左边
    lhs, rhs = sympy.simplify(eq.lhs - eq.rhs), 0
    
    # 提取左边的项
    terms = lhs.as_ordered_terms()
    
    # 存储加项和减项
    positive_terms = []
    negative_terms = []
    
    for term in terms:
        if term.is_negative:
            # 将减项存储
            negative_terms.append(-term)
        else:
            # 将加项存储
            positive_terms.append(term)
    
    # 组合成新方程
    new_lhs = sympy.Add(*positive_terms)
    new_rhs = sympy.Add(*negative_terms)
    
    # 生成字符串输出
    result = f"{new_lhs} = {new_rhs}"
    return result




# Tools for generator

def get_points_mapping(clause, template):
    clause_content = re.findall(r'\((.*?)\)', clause)
    template_content = re.findall(r'\((.*?)\)', template)

    # 检查是否找到了匹配项并且确保两者都只有一个括号对
    if len(clause_content) != 1 or len(template_content) != 1:
        raise ValueError("Invalid input strings, expected exactly one pair of parentheses.")

    # 分割出括号内的各个元素
    clause_elements = clause_content[0].replace(',', '')
    template_elements = template_content[0].replace(',', '')

    # 确保两个字符串的括号内元素数量一致
    if len(clause_elements) != len(template_elements):
        raise ValueError("The number of elements within parentheses must match.")

    # 创建映射
    mapping = {}
    for t, c in zip(template_elements, clause_elements):
        mapping[t.strip()] = c.strip()

    return mapping

def get_symbol(char):
    return [Symbol(f"x_{char}"), Symbol(f"y_{char}")]

def get_content(clause):
    if ',' in clause:
        match = re.search(r'\((.*?)\)', clause)
        items = match.group(1)
        items = items.split(',')
        points = list(set(sum([list(s.strip()) for s in items], [])))
    else:
        items = re.findall(r'\((\w+)\)', clause)
        points = sum((list(set(item)) for item in items), [])
    return sorted(points)
        
    
def get_points_num(clause):
    if ',' in clause:
        match = re.search(r'\((.*?)\)', clause)
        items = match.group(1)
        items = items.split(',')
        points = list(set(sum([list(s.strip()) for s in items], [])))

    else:
        items = re.findall(r'\((\w+)\)', clause)
        points = sum((list(set(item)) for item in items), [])
    return len(points)

def replace_points(predicate, pred_info, new_points, mapping=None):
    # Predicate(* Points) -> Predicate(* New Points) 
    # Given point mapping: do not use new_points
    if mapping == None:
        mapping = {}
        points = get_content(predicate)
        assert len(points) == len(new_points)
        for p, new_p in zip(points, new_points):
            mapping[p] = new_p
        
    def replace_for_predicate(predicate, p_map):
        if 'Equal' in predicate:
            items = predicate.split('Equal(')[-1][:-1]
            left, right = items.split(',')
            left = replace_for_predicate(left, p_map)
            right = replace_for_predicate(right, p_map)
            return f'Equal({left},{right})'
        elif '(' not in predicate and ')' not in predicate:
            return predicate
        else:
            name, letter_part = predicate.split('(')
            letter_part = letter_part.rstrip(')')
            letter_part = ''.join(p_map.get(c, c) for c in letter_part)
            return f"{name}({letter_part})"

    predicate = replace_for_predicate(predicate, mapping)
    for i, pred_i in enumerate(pred_info['ee_check']):
        pred_info['ee_check'][i] = replace_for_predicate(pred_i, mapping)
    for i, pred_i in enumerate(pred_info['extend']):
        pred_info['extend'][i] = replace_for_predicate(pred_i, mapping)

    return predicate, pred_info


def replace_for_clause(predicate, p_map):
    if 'Equal' in predicate:
        items = predicate.split('Equal(')[-1][:-1]
        left, right = items.split(',')
        left = replace_for_clause(left, p_map)
        right = replace_for_clause(right, p_map)
        return f'Equal({left},{right})'
    elif '(' not in predicate and ')' not in predicate:
        return predicate
    else:
        name, letter_part = predicate.split('(')
        letter_part = letter_part.rstrip(')')
        letter_part = ''.join(p_map.get(c, c) for c in letter_part)
        return f"{name}({letter_part})"



def append_lst(lst, items: list):
    for item in items:
        if item not in lst:
            lst.append(item)
            
    return lst

def parse_clause(clause):
    if 'Value' in clause:
        clause = clause.replace('Value', 'Equal')
    if 'Equal' in clause:
        pattern =  r'Equal\((.*)\)'
        match = re.search(pattern, clause)
        items = match.group(1)
        item_l, item_r = items.split(',')
        predicate_l, predicate_r = None, None
        content_l, content_r = item_l, item_r
        if not item_l.isdigit() and item_l.split('(')[0] in PREDICATES:
            predicate_l, content_l = parse_clause(item_l)
            if len(content_l) == 1:
                content_l = content_l[0]
        if not item_r.isdigit() and item_r.split('(')[0] in PREDICATES:
            predicate_r, content_r = parse_clause(item_r)
            if len(content_r) == 1:
                content_r = content_r[0]
        content_l = content_l.strip()
        content_r = content_r.strip()
        predicate_l = predicate_l.strip()
        if predicate_r is None or predicate_r == predicate_l:
            return predicate_l, (content_l, content_r)
        else:
            print(predicate_r)
            print('Error')
    elif '(' not in clause and ')' not in clause:
        return clause
    else:
        # "Equation(ll_ac-sqrt(2)*ll_bd)"
        items = '('.join(clause.split('(')[1:])[:-1]
        items = [i.strip() for i in items.split(',')]
        predicate = clause.split('(')[0].strip()
        
        return predicate, items
    
def get_points(clause):
    _, items = parse_clause(clause)
    points = set()
    for item in items:
        if item.isdigit():
            continue
        if len(item) > 1:
            points.update(list(item))
        else:
            points.add(item)
            
    return sorted(list(points))
    
def max_letter_index(s):
    # index 0: letter idx
    # index 1: others > Equal > Cocircular
    if 'Equal' in s:
        pattern =  r'Equal\((.*)\)'
        match = re.search(pattern, s)
        if match:
            # 提取匹配到的内容
            items = match.group(1)
            max_index_l, max_index_r = (0, 0), (0, 0)
            item_l, item_r = items.split(',')
            if not item_l.isdigit(): 
                max_index_l = max_letter_index(item_l)
            if not item_r.isdigit():
                max_index_r = max_letter_index(item_r)
            max_index = max(max_index_l, max_index_r)
            return (max_index[0], 1)
        else:
            return (0, 1)
    else:
        content = re.findall(r'\((.*?)\)', s)[0]
        # 计算括号内每个字母的索引，并找出最大索引
        index = sorted([ord(c) - ord('a') for c in content if c.isalpha()], reverse=True)
        
        index_0 = index[0]
        index_1 = 0 
        if 'Cocircular' in s: # move cocircular to the last
            index_1 = 2
        return (index_0, index_1)
    
def find_target_for_construct(clause):
    # 返回需要构造的target points
    # 以其他点为条件，可确定target points的坐标
    predicate, items = parse_clause(clause)
    if predicate in ['Shape', 'Triangle'] :
        return []
    if predicate == 'Polygon':
        if len(items[0]) <= 3:
            return []
        if len(items[0]) == 4:
            return [max(items[0])]
            
    if predicate == 'LengthOfLine':
        return [max(items[0] + items[1])]
    if predicate == 'LengthOfArc':
        points_1 = items[0][1:]
        points_2 = items[1][1:]
        return [max(points_1 + points_2)]
    elif predicate == 'MeasureOfAngle':
        a1, a2 = items
        if a2.isdigit():
            return [max(a1)]
        else:
            return [max(a1 + a2)]
    elif predicate == 'Collinear':
        points = sorted(items[0])
        return points[2:]
    elif predicate == 'Cocircular':
        circle, points = items
        sorted_points = sorted(points)
        if circle > max(sorted_points[:3]):
            return [circle] + list(points[3:])
        else:
            return list(sorted_points[1:])
    elif predicate == 'ParallelBetweenLine':
        return [max(items[0] + items[1])]
    elif predicate in ['SimilarBetweenTriangle', 
                       'SimilarBetweenQuadrilateral',
                       'CongruentBetweenTriangle', 
                       'CongruentBetweenQuadrilateral']:
        # 对于全等和相似，需要构造所有的点
        return list(items[0]) + list(items[1])
    elif predicate == 'IsCircumcenterOfQuadrilateral':
        return items[0]
    elif predicate == 'IsCentroidOfTriangle':
        return items[0]
    else:
        raise KeyError(predicate)

def intersects(line1, line2):
    """检查两条线段是否相交"""
    return bool(set(line1) & set(line2))

def setup_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)

def simplify_and_trim(poly, threshold=1e-5):
    poly = simplify(poly)
     # 将多项式中的浮点数转换为 SymPy 的 Float 类型
    poly = poly.xreplace({n: Float(n) for n in poly.atoms(float)})
    
    # 获取多项式的系数
    coeffs = poly.as_coefficients_dict()
    
    # 创建一个新的系数字典，将接近零的系数替换为零
    new_coeffs = {key: 0 if abs(value.evalf()) < threshold else value for key, value in coeffs.items()}
    
    # 构建新的多项式
    new_poly = sum(value * key for key, value in new_coeffs.items())
    
    # 返回简化后的多项式
    return new_poly

def remove_duplicates(lst):
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result

def random_generate_line_length():
    flag = random.choice([0, 1])
    if flag == 0:   # 2x+1
        num1 = random.choice(list(range(1, 10)))
        num2 = random.choice(list(range(1, 10)))
        symbol = Symbol(random.choice(['x', 'y', 'z', 'a', 'b', 'c']))
        add_flag = random.choice([True, False])
        if add_flag:
            expr = num1 * symbol + num2
        else:
            expr = num1 * symbol - num2
        res_string = str(expr)
    elif flag == 1: # 2
        num = random.choice(list(range(1, 20)))
        res_string = str(num)
    
    res_string = res_string.replace(' ', '').replace('*', '')
    return res_string

def random_generate_angle_measure(p_cross, p1, p2):
    flag = random.choice([0, 1])
    if flag == 0: # x
        res_string = random.choice(['x', 'y', 'z', 'a', 'b', 'c'])
    elif flag == 1: # 2
        num = get_angle_measure(p_cross, p1, p2)
        res_string = str(num)
        
    res_string = res_string.replace(' ', '').replace('*', '')
    return res_string

def get_angle_measure(p_cross, p1, p2):
    # Calculate the vectors
    v1 = (p_cross[0] - p1[0], p_cross[1] - p1[1])
    v2 = (p_cross[0] - p2[0], p_cross[1] - p2[1])

    # Calculate the dot product
    dot = v1[0] * v2[0] + v1[1] * v2[1]

    # Calculate the magnitudes
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # Calculate the cosine of the angle
    cos_angle = dot / (mag_v1 * mag_v2)

    # Calculate the angle in radians
    angle_rad = math.acos(cos_angle)

    # Convert to degrees
    angle_deg = round(angle_rad * 180 / math.pi)
    assert angle_deg <= 180 and angle_deg >= 0
    return angle_deg

def formalgeo_to_intergps(clause):
    name, items = parse_clause(clause)
    logic_form_list = []
    if name in ['Triangle', 'Parallelogram', 'Rectangle', 'Rhombus', 'Square', 'Trapezoid', 'Kite']:
        points = list(items[0])
        logic_form = f"{name}({','.join(points)})"
        logic_form_list.append(logic_form)
        
    if name == 'Collinear':
        points = list(items[0])
        assert len(points) == 3
        p1, p2, p3 = points
        logic_form = f"PointLiesOnLine({p2},Line({p1},{p3}))"
        logic_form_list.append(logic_form)
        
    elif name == 'Cocircular':
        circle, points = items
        for p in points:
            logic_form = f"PointLiesOnCircle({p},Circle({circle}))"
            logic_form_list.append(logic_form)
            
    elif name == 'RightTriangle':
        p1, p2, p3 = items[0]
        logic_form_list += [
            f"Polygon({','.join(items[0])})",
            f"Perpendicular(Line({p1},{p2}),Line({p2},{p3}))"
        ]
        
    elif name == 'EquilateralTriangle':
        points = list(items[0])
        logic_form = f"Equilateral(Triangle({','.join(points)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsoscelesTriangle':
        points = list(items[0])
        logic_form = f"Isosceles(Triangle({','.join(points)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'RightTrapezoid':
        points = list(items[0])
        p1, p2, p3, p4 = points
        # DA \perp BA, AB \perp CB
        logic_form_list += [
            f"Trapezoid({','.join(points)})",
            f"Perpendicular(Line({p4},{p1}),Line({p2},{p1}))",
            f"Perpendicular(Line({p1},{p2}),Line({p3},{p2}))",
        ]
        
    elif name == 'IsoscelesTrapezoid':
        points = list(items[0])
        logic_form = f"Isosceles(Trapezoid({','.join(points)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsoscelesRightTriangle':
        points = list(items[0])
        p1, p2, p3 = points
        logic_form_list += [
            f"Polygon({','.join(points)})",
            f"Perpendicular(Line({p1},{p2}),Line({p2},{p3}))",
            f"Isosceles(Triangle({','.join(points)}))"
        ]

    elif name == 'PerpendicularBetweenLine':
        l1, l2 = items
        logic_form = f"Perpendicular(Line({','.join(l1)}),Line({','.join(l2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'ParallelBetweenLine':
        l1, l2 = items
        logic_form = f"Parallel(Line({','.join(l1)}),Line({','.join(l2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsTangentOfCircle':
        line, circle = items
        logic_form = f"Tangent(Line({','.join(line)}),Circle({circle}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsDiameterOfCircle':
        line, circle = items
        logic_form = f"IsDiameterOf(Line({','.join(line)}),Circle({circle}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsMidpointOfLine':
        mid_p, line = items
        logic_form = f"IsMidpointOf(Point({mid_p}),Line({','.join(line)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsBisectorOfAngle':
        line, angle = items
        logic_form = f"BisectsAngle(Line({','.join(line)}),Angle({','.join(angle)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsPerpendicularBisectorOfLine':
        l1, l2 = items
        logic_form = f"IsPerpendicularBisectorOf(Line({','.join(l1)}),Line({','.join(l2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsAltitudeOfTriangle':
        line, tri = items
        logic_form = f"IsAltitudeOf(Line({','.join(line)}),Triangle({','.join(tri)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsMedianOfTriangle':
        line, tri = items
        logic_form = f"IsMedianOf(Line({','.join(line)}),Triangle({','.join(tri)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsIncenterOfTriangle':
        p, tri = items
        logic_form = f"IsIncenter(Point({p}),Triangle({','.join(tri)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsMidsegmentOfTriangle':
        line, tri = items
        logic_form = f"IsMidsegmentOf(Line({','.join(line)}),Triangle({','.join(tri)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsMidpointOfArc':
        p0, arc = items
        c, p1, p2 = arc
        logic_form = f"Equals(MeasureOf(Arc({p0},{p1})),MeasureOf(Arc({p1},{p2})))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsMidsegmentOfQuadrilateral':
        line, quad = items
        logic_form = f"IsMidsegmentOf(Line({','.join(line)}),Quadrilateral({','.join(quad)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'IsCentroidOfTriangle':
        p, tri = items
        logic_form = f"IsCentroidOf(Point({p}),Triangle({','.join(tri)}))"
        logic_form_list.append(logic_form)
        
    elif name in ['IsCircumcenterOfTriangle', 'IsCircumcenterOfQuadrilateral']:
        circle, points = items
        for p in points:
            logic_form = f"PointLiesOnCircle({p},Circle({circle}))"
            logic_form_list.append(logic_form)
            
    elif name == 'SimilarBetweenTriangle':
        poly1, poly2 = items
        logic_form = f"Similar(Triangle({','.join(poly1)}),Triangle({','.join(poly2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'SimilarBetweenQuadrilateral':
        poly1, poly2 = items
        logic_form = f"Similar(Quadrilatera({','.join(poly1)}),Quadrilatera({','.join(poly2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'CongruentBetweenTriangle':
        poly1, poly2 = items
        logic_form = f"Congruent(Triangle({','.join(poly1)}),Triangle({','.join(poly2)}))"
        logic_form_list.append(logic_form)
        
    elif name == 'CongruentBetweenQuadrilateral':
        poly1, poly2 = items
        logic_form = f"Congruent(Quadrilateral({','.join(poly1)}),Quadrilateral({','.join(poly2)}))"
        logic_form_list.append(logic_form)
    
    return logic_form_list
    
    
if __name__ == '__main__':
    from inter_gps_solver.extended_definition import ExtendedDefinition
    from inter_gps_solver.logic_parser import LogicParser
    data = json.load(open('json/predicate_GDL.json', 'r'))
    for clause in data['Entity']:
        name, _ = parse_clause(clause)
        if name not in PREDICATES_ENT:
            continue
        logic_forms = formalgeo_to_intergps(clause)
        print(clause + ' -> ' + ', '.join(logic_forms))
        parser = LogicParser(ExtendedDefinition(debug=True))
        parser.logic.point_positions = None
        for logic_form in logic_forms:
            parse_tree = parser.parse(logic_form)
            parser.dfsParseTree(parse_tree)
    
    for clause in data['Relation']:
        name, _ = parse_clause(clause)
        if name not in PREDICATES_REL:
            continue
        logic_forms = formalgeo_to_intergps(clause)
        print(clause + ' -> ' + ', '.join(logic_forms))
        parser = LogicParser(ExtendedDefinition(debug=True))
        parser.logic.point_positions = None
        for logic_form in logic_forms:
            parse_tree = parser.parse(logic_form)
            parser.dfsParseTree(parse_tree)
    
