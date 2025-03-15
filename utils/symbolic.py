import math
import random
import re

import sympy
from sympy import Add, Float, Mul, Symbol, UnevaluatedExpr, simplify, preorder_traversal

from .preset import PREDICATES


def parse_expression(expr):
    expr = expr.strip()
    if '(' in expr and expr.endswith(')'):
        name_part, rest = expr.split('(', 1)
        name = name_part.strip()
        args_str = rest[:-1]
        args = split_args(args_str)
        parsed_args = []
        for arg in args:
            parsed_arg = parse_expression(arg)
            parsed_args.append(parsed_arg)
        return {'name': name, 'args': parsed_args}
    else:
        return list(expr)

def split_args(args_str):
    args = []
    current = []
    stack = []
    for c in args_str:
        if c == '(':
            stack.append('(')
        elif c == ')':
            if stack:
                stack.pop()
        elif c == ',' and not stack:
            args.append(''.join(current).strip())
            current = []
            continue
        current.append(c)
    if current:
        args.append(''.join(current).strip())
    return args

def collect_mappings(source, target, mappings):
    if isinstance(source, dict) and isinstance(target, dict):
        if source['name'] != target['name']:
            return False
        if len(source['args']) != len(target['args']):
            return False
        for s_arg, t_arg in zip(source['args'], target['args']):
            if not collect_mappings(s_arg, t_arg, mappings):
                return False
        return True
    elif isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            return False
        for s_elem, t_elem in zip(source, target):
            if not collect_mappings(s_elem, t_elem, mappings):
                return False
        return True
    elif isinstance(source, str) and isinstance(target, str):
        if len(source) != 1 or len(target) != 1:
            return False
        t_char = target
        s_char = source
        if t_char in mappings:
            if mappings[t_char] != s_char:
                return False
        else:
            mappings[t_char] = s_char
        return True
    else:
        return False

def build_point_map(source_clause, target_clause):
    try:
        source_tree = parse_expression(source_clause)
        target_tree = parse_expression(target_clause)
    except:
        return None
    
    mappings = {}
    if not collect_mappings(source_tree, target_tree, mappings):
        return None
    
    values = list(mappings.values())
    if len(values) != len(set(values)):
        return None
    
    return mappings

def replace_points_for_clause(clause, p_map):  
    pattern = r'(?<=[(,])[A-Z]+(?=[,)])'
    
    def repl(match):
        # match.group(0) 获取匹配到的变量（例如 "AC"）
        var = match.group(0)
        # 对变量中的每个字母进行替换
        return ''.join(p_map.get(letter, letter) for letter in var)
    
    # 使用 re.sub 进行替换
    return re.sub(pattern, repl, clause)

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

def replace_points(predicate, pred_info=None, new_points=None, mapping=None):
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
    if pred_info is not None:
        for i, pred_i in enumerate(pred_info['ee_check']):
            pred_info['ee_check'][i] = replace_for_predicate(pred_i, mapping)
        for i, pred_i in enumerate(pred_info['extend']):
            pred_info['extend'][i] = replace_for_predicate(pred_i, mapping)

    return predicate, pred_info

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

def get_predicate_name(clause):
    if 'Equal' in clause:
        return 'Equal'
    else:
        return clause.split('(')[0]
    
def get_symbol(char):
    return [Symbol(f"x_{char}"), Symbol(f"y_{char}")]

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
        if predicate_l is not None:
            predicate_l = predicate_l.strip()
            if predicate_r is None or predicate_r == predicate_l:
                # Equal(MeasureOfAngle, MeasureOfAngle)
                return predicate_l, (content_l, content_r)
            elif predicate_r is not None: 
                # Equal(MeasureOfAngle, MeasureOfArc)
                return 'Equal', (item_l, item_r)
        else: 
            # Equal(y, 10)
            return 'Equal', (item_l, item_r)
        
    elif '(' not in clause and ')' not in clause:
        return clause
    else:
        # "Equation(ll_ac-sqrt(2)*ll_bd)"
        items = '('.join(clause.split('(')[1:])[:-1]
        items = [i.strip() for i in items.split(',')]
        predicate = clause.split('(')[0].strip()
        
        return predicate, items

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

def random_generate_angle_measure(p_cross, p1, p2):
    flag = random.choice([0, 1])
    if flag == 0: # x
        res_string = random.choice(['x', 'y', 'z', 'a', 'b', 'c'])
    elif flag == 1: # 2
        num = get_angle_measure(p_cross, p1, p2)
        res_string = str(num)
        
    res_string = res_string.replace(' ', '').replace('*', '')
    return res_string

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
    
def subs_without_simplification(expr, substitutions):
    for old, new in substitutions.items():
        expr = expr.subs(old, UnevaluatedExpr(new))
    return expr

def degree_of_expr(expr):
    try:
        return expr.as_poly(*list(expr.free_symbols)).total_degree()
    except:
        return 3 # ignore

