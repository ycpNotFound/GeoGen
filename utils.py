import json
import re
from collections import Counter
from sympy import Symbol
import random
import numpy as np
import sympy
from sympy import Float, Add, Mul, simplify
from matplotlib import pyplot as plt

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
PRESET_COLOR_PROBS = [0.6, 0.08, 0.08, 0.08, 0.08, 0.08]
PRESET_COLORS = [
    # line | point | char | annoation | fill_in
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


def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def load_theorems_and_predicates():
    path_1 = "doc/predicates.txt"
    path_2 = "doc/theorems.txt"
    
    with open(path_1, 'r') as f1:
        preds = f1.readlines()
        preds = [s.strip() for s in preds]
        
    with open(path_2, 'r') as f2:
        theos = f2.readlines()
        theos = [s.strip() for s in theos]
        
    return preds, theos

def stats_for_formalgeo():
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = read_json(data_path)
    preds, theos = load_theorems_and_predicates()
    
    cdl_list, theo_list = [], []
    for key, value in data.items():
        cdl_list_i = value['construction_cdl'] + value['text_cdl']
        
            
        cdl_list_1 = [s.split('(')[0] for s in cdl_list_i]
        cdl_list_2 = []
        for s in cdl_list_i:
            if 'Equal' in s:
                s = s.split('Equal(')[-1][:-1]
                s = s.split('(')[0]
                if s[0].isupper():
                    cdl_list_2.append(s)
        
        cdl_list_i = cdl_list_1 + cdl_list_2
        theo_list_i = value['theorem_seqs']
        theo_list_i = [s.split('(')[0] for s in theo_list_i]
        
        cdl_list += cdl_list_i
        theo_list += theo_list_i
    
    cdl_count = Counter(cdl_list)
    theo_count = Counter(theo_list)
    
    preds_not_used = set(preds) - set(list(cdl_count.keys()))
    print(f"{'-'*10} Predicats not used : {'-'*10}")
    for p in preds_not_used:
        print(p)
        
    theo_not_used = set(theos) - set(list(theo_count.keys()))
    print("{'-'*10} Theorems not used: {'-'*10}")
    for t in theo_not_used:
        print(t)
        
    pred_sorted = sorted(list(cdl_count.keys()), key=lambda x: cdl_count[x], reverse=True)
    print(f"{'-'*10} Predicats Stats: {'-'*10}")
    for p in pred_sorted:
        print(f"{cdl_count[p]} - {p}")
        
    theo_sorted = sorted(list(theo_count.keys()), key=lambda x: theo_count[x], reverse=True)
    print(f"{'-'*10} Theorem Stats: {'-'*10}")
    for t in theo_sorted:
        print(f"{theo_count[t]} - {t}")
        
    return cdl_count, theo_count


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

def append_lst(lst, items: list):
    for item in items:
        if item not in lst:
            lst.append(item)
            
    return lst

def parse_clause(clause):
    if 'Equal' in clause:
        pattern =  r'Equal\((.*)\)'
        match = re.search(pattern, clause)
        items = match.group(1)
        item_l, item_r = items.split(',')
        predicate_l, predicate_r = None, None
        content_l, content_r = item_l, item_r
        if not item_l.isdigit():
            predicate_l, content_l = parse_clause(item_l)
            if len(content_l) == 1:
                content_l = content_l[0]
        if not item_r.isdigit():
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
    else:
        items = re.findall(r'\((.*?)\)', clause)[0]
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
    
if __name__ == '__main__':
    # stats_for_formalgeo()
    s = "Cocircular(O,ABC)"
    a = get_content(s)
    print(a)