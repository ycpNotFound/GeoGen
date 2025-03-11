from sympy import simplify, nsimplify, Integer, Float, symbols, linear_eq_to_matrix, solve, Eq, Matrix
import re
import os
import numpy as np
from formalgeo_v2.core.engine import EquationKiller as EqKiller
import requests
import json
import itertools
from tqdm import tqdm

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
    
def test_print():
    a =  "\\arc {p1[1:]} is congruent to \\arc {p2[1:]}"
    a = a.format(p1=list('xcd'), p2=list('xad'))
    print(a)
    
def test_func_timeout():
    from func_timeout import func_timeout, FunctionTimedOut

    def my_function(param1, param2):
        # 假设这是一个可能会运行很长时间的函数
        while True:
            pass

    try:
        # 尝试运行 my_function 5 秒钟超时
        result = func_timeout(2, my_function, args=('value1', 'value2'))
    except FunctionTimedOut as e:
        # 自定义异常处理逻辑，这里可以选择性地打印或记录日志
        print("The function took too long to execute.")
        # 如果你不希望任何信息被打印，可以简单地pass
        # pass
        
def test_chat():
    from main_chat import Agent
    agent = Agent('api.json')
    model_name = 'qwen2.5-7b-instruct'
    instruction = """## Transcription task
Convert the following structured information parsed from a geometry diagram into a natural language description. 
Output Requirements: 
- The output should be a coherent piece of natural language text.
- Do not miss geometry information, such as points, lines, circles(if exist), length or angle measure, and so on.
- Try to keep description concise and mathematical, and avoid reasoning.
""" 
# - Try to refer to geometry knowledge related to the diagram, while keep description concise. """
    input_str = [
        "The geometric construction in this figure includes angle bisector, equilateral triangle and tangent line. Here is a detailed description:\nPoints: $ I, J, K,L,M,N $ .\nLines: $ I,JK,IK,KL,IN,IM,IM,NLK $ .\ncircles: $M$.\nGeometry Relations:\n- The shape IIK forms an equilateral triangle.\n- KL is the besector line of angle IkJ.\n- NI is the tangent of circle M.\nangle IJM = (4c+2)°.\n-Points N,L,K lie on the same line.\n-Points I,J,K,L lie on the circle M.",
        
        "In this diagram, the geometric pattern consists of triangle. Here is a detailed description:\nPoints: $ A, B, C $.\nLines: $ AB, BC, AC $ .\ncircles: there's no circle in the diagram. \n Geometry Relations:\n- The shape ABC forms a triangle. \n- ∠BAC =60°.\n- CAB =(3a-4)°",
        
        "The diagram shows a geometric figure with parallelogram. Here is a detailed description:\nPoints: $ I, J, K, L $.\nLines: $IJ, JK, KL, IL$.\nCircles: there's no circle in the diagram.\nGeometry Relations:\n- IJKL is a parallelogram."
    ]
    for str_i in input_str:
        output, _ = agent.chat(f"{instruction}\n{str_i}", model_name)
        print('-------------------------------')
        print(output)
        print(_)
        
    a = 1


def find_formal_in_problem_text():
    import os
    from utils.preset import PREDICATES, PREDICATES_ENT
    train_dir = 'datasets/processed_data/fgo_train'
    delete_names = [p for p in PREDICATES if p not in PREDICATES_ENT]
    cnt = 0
    correct_paths = []
    for name in os.listdir(train_dir):
        data = json.load(open(f"{train_dir}/{name}", 'r', encoding='utf-8'))

        problem = data['llm_info']['problem_text']
        if any([x in problem for x in delete_names]):
            if data['solved']:
            # print(problem)
                print(f"{train_dir}/{name}")
                correct_paths.append(f"{train_dir}/{name}")
                cnt += 1
                
    print(cnt)
    return correct_paths
    

def test_itertools():
    """测试使用itertools进行数据批处理。
    
    将一个迭代器按照指定进程数均匀分割成多个批次。
    """
    import itertools

    # 定义常量
    TOTAL_ITEMS = 100
    NUM_PROCESSES = 10
    
    # 创建测试数据
    test_iterator = range(TOTAL_ITEMS)
    batch_size = TOTAL_ITEMS // NUM_PROCESSES
    
    # 按批次分割数据
    batches = []
    for process_id in range(NUM_PROCESSES):
        start_idx = process_id * batch_size
        end_idx = (process_id + 1) * batch_size
        batch = itertools.islice(test_iterator, start_idx, end_idx)
        batches.append(batch)
    
    # 打印每个批次的内容
    for batch in batches:
        print(list(batch))
        
def test_template():
    from utils.formulate import clause_to_nature_language
    from utils.tools import read_json
    template = read_json('json/predicates_to_nature_language.json')
    clause = 'Equal(LengthOfLine(AB),RadiusOfCircle(O))'
    res = clause_to_nature_language(clauses=[clause],
                                    natural_template=template)
    print(res)
        
def test_subs():
    from itertools import product
    def substitude_equivalene(expr, premise_ids, values_dict):
        variables = list(expr.free_symbols)
        value_combinations = list(product(*[values_dict[var] for var in variables]))
        substitude_res = []
        for values_group in value_combinations:
            values = [item[0] for item in values_group]
            premises = sum([item[1] for item in values_group], [])
            subs_dict = dict(zip(variables, values))
            new_expr = expr.subs(subs_dict)
            substitude_res.append((new_expr, premise_ids + premises))
    
        return substitude_res
    
    from sympy import Symbol

    a, b, c, x, y, p, q, m, n = symbols('a b c x y p q m n')
    expr = a + b + c
    premise_ids = [0]
    values_dict = {
        a: [(x, [1]), (y, [2])],
        b: [(p, [3]), (q, [4])],
        c: [(m, [5]), (n, [6])]
    }
    res = substitude_equivalene(expr, premise_ids, values_dict)
    print(res)

def test_adjust_coef():
    from sympy import Add, Mul, Number, symbols
    def adjust_first_variable_coefficient(expr):
        """首先获取expr的第一个变量，然后将第一个变量的系数调整为1"""
        if not isinstance(expr, Add):
            return expr
        args = expr.args
        def index_arg_by_chars(arg):
            free_symbols = arg.free_symbols
            if len(free_symbols) == 0:
                return 'z'
            else:
                return str(sorted(free_symbols)[0])
        sorted_args = sorted(args, key=lambda x: index_arg_by_chars(x))
        first_expr = sorted_args[0]

        coeff = first_expr.as_coeff_Mul()[0]
        if coeff is None:
            coeff = 1

        adjusted_expr = expr * (1 / coeff)
        return adjusted_expr

    x, y, z = symbols('x y z')
    expr_1 = x + y + z
    expr_2 = - x**2 + y**2 + z**2
    expr_3 = x - 2*y + 1
    expr_4 = -2* x - 2 * y - 1
    for expr in [
        # expr_1, expr_2, 
        expr_3, expr_4
    ]:
        print('=====================')
        print(adjust_first_variable_coefficient(expr))
        print(expr)

def test_check_expr():
    import sympy
    from sympy import degree, preorder_traversal, Add, Pow, symbols
    def contains_sqrt_once(expr):
        """检查表达式中是否仅有一个子表达式包含根号"""
        sqrt_count = 0
        for node in preorder_traversal(expr):
            if isinstance(node, sympy.Pow) and node.exp == sympy.S.Half:
                sqrt_count += 1
            # 如果超过一个根号，则直接返回False
            if sqrt_count > 1:
                return False
        return sqrt_count == 1
    def check_expr(expr, min_symbol_num=2, max_items_num=3):
        # 筛选符合条件的expr: ax + by + c = 0
        free_symbols = expr.free_symbols
        
        # free symbols num最多为2
        if len(free_symbols) not in list(range(min_symbol_num, 3)):
            return False
        
        # 最大项数为max_items_num
        terms = expr.args if expr.is_Add else (expr,)
        if len(terms) > max_items_num:
            return False
        
        # 最大次数为1或2
        max_degree = 0
        for symbol in free_symbols:
            current_degree = degree(expr, symbol)
            if current_degree > max_degree:
                max_degree = current_degree
        if max_degree == 1:
            # 允许一个根号
            contain_sqrt = contains_sqrt_once(expr)
            if contain_sqrt:
                # 检查根号内的部分是否为常数
                for node in preorder_traversal(expr):
                    if isinstance(node, sympy.Pow) and node.exp == sympy.S.Half:
                        sqrt_base = node.args[0]  # 获取根号内的表达式
                        if not sqrt_base.is_number:  # 如果根号内的表达式不是常数
                            return False
                    
        # 如果为2则必须类似：ax^2 - by^2, ax^2 - by, ax - by^2
        elif max_degree == 2:
            if contains_sqrt_once(expr):
                return False
            if not expr.is_Add:
                return False
            if len(expr.args) != 2:
                return False
            if expr.as_coefficients_dict().get(1, 0) != 0:
                return False
        
        return True
    
    x, y = symbols('x y')
    # expr_1 = x + y
    expr_2 = x**2 - y**2
    expr_3 = x**2 - y**2 + 1
    expr_4 = x**2 - y**2 + x
    for expr in [expr_2, expr_3, expr_4]:
        print(check_expr(expr))

def test_formulate():
    from utils.formulate import parse_clause
    from utils.formulate import clause_to_nature_language
    from utils.tools import read_json
    template = read_json('json/predicates_to_nature_language.json')
    clauses = [
        # 'Equal(LengthOfLine(AB),2)',
        # 'Equal(MeasureOfAngle(ABC),90)',
        # 'Value(MeasureOfArc(KLJ),148)',
        'Value(PerimeterOfCircle(A),20*pi)',
        'Equal(MeasureOfAngle(ABC),MeasureOfArc(OAC))',
        'Equal(MeasureOfAngle(ABC),MeasureOfArc(OAC))',
        # 'Equal(LengthOfLine(AB),RadiusOfCircle(O))',
        # 'Equal(LengthOfLine(AB),LengthOfArc(OAC))',
        # 'Equal(LengthOfLine(AB),Mul(1/2,LengthOfArc(OAC)))',
        # 'Equal(LengthOfLine(AB),Mul(sqrt(2),LengthOfArc(OAC)))',
        'Equal(Add(LengthOfLine(AB),LengthOfLine(AC)),Mul(sqrt(2),LengthOfArc(OAC)))',
        'Equal(x,10)'
    ]
    
    res = clause_to_nature_language(clauses=clauses,
                                    natural_template=template)
    
    for res_i in res:
        print(res_i)

if __name__ == '__main__':

    
    # test_subs()
    # test_adjust_coef()
    # test_check_expr()
    # test_formulate()
    dir_1 = 'datasets/fgo_search_train_v2'
    dir_2 = 'datasets/fgo_search_train_v3'
    files_1 = os.listdir(dir_1)
    files_2 = os.listdir(dir_2)

    import shutil
    cnt = 0
    for f_in_1 in tqdm(files_1):
        if not os.path.exists(f"{dir_2}/{f_in_1}"):
            cnt += 1
            shutil.copy2(f"{dir_1}/{f_in_1}", f"{dir_2}/{f_in_1}")

    print(cnt)