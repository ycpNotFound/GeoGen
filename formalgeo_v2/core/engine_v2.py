import copy
import time
# from formalgeo.problem import Problem
import warnings
from itertools import combinations, product

import sympy
from func_timeout import FunctionTimedOut, func_set_timeout
from sympy import (Add, Float, Integer, Mul, Number, Poly, Pow, Rational, Expr,
                   Symbol, degree, total_degree, preorder_traversal, simplify, solve, sqrt,
                   symbols)

from formalgeo_v2.parse import get_equation_from_tree
from formalgeo_v2.tools import rough_equal


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

def variable_in_function(expr):
    # 获取表达式中的所有符号（变量）
    variables = expr.free_symbols  
    if not variables:  
        return False  # 没有符号变量，直接返回 False
    
    # 检查复杂函数的参数是否包含变量
    vars_in_func = False
    for sub_expr in preorder_traversal(expr):
        if sub_expr.is_Function:  # 只考虑函数对象
            if any(arg.has(*variables) for arg in sub_expr.args):  # 其参数是否含变量
                vars_in_func = True
                break
    return vars_in_func

def check_expr(expr, min_symbol_num=2, max_items_num=3):
    # 筛选符合条件的expr: ax + by + c = 0
    free_symbols = expr.free_symbols

    if 'sin' in str(expr) or 'cos' in str(expr):
        return False
    
    # free symbols num最多为2
    if len(free_symbols) not in list(range(min_symbol_num, 3)):
        return False
    
    # 最大项数为max_items_num
    terms = expr.args if expr.is_Add else (expr,)
    if len(terms) > max_items_num:
        return False
    
    # 最大次数为1或2
    if 'aq_hgfj' in str(expr):
        a = 1
    max_degree = total_degree(expr)
    
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
    # 如果是面积 / 周长 / 高，只允许出现一次
    special_var_prefix = ['pt_', 'at_', 'ht_', 'pq_', 'aq_', 'hq_', 'pc_', 'ac_', 'ps_', 'as_']
    special_vars = [
        sym for sym in list(expr.free_symbols)
        if any(p in str(sym) for p in special_var_prefix)
    ]
    if len(special_vars) >= 2 and max_degree >= 2:
        return False
        
    return True

def check_expr_perimeter_area(expr):
    special_var_prefix = ['pt_', 'at_', 'pq_', 'aq_', 'pc_', 'ac_', 'ps_', 'as_']
    special_vars = [
        sym for sym in list(expr.free_symbols)
        if any(p in str(sym) for p in special_var_prefix)
    ]
    if len(special_vars) >= 2:
        return False
        
    return True

def extract_coefficient(expr):
    # 获取表达式中的符号集合
    symbols = expr.free_symbols
    if len(symbols) != 1:
        raise ValueError("表达式必须只含有一个符号")
    sym = symbols.pop()
    
    # 如果表达式直接等于符号本身，系数为1
    if expr == sym:
        return 1
    
    # 处理乘积表达式
    if isinstance(expr, Mul):
        coeff = 1
        for factor in expr.args:
            if not factor.has(sym):
                coeff *= factor
        return coeff
    else:
        # 对于非乘积类型，如x**2，只要包含符号则系数为1
        return 1 if expr.has(sym) else expr

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
            return str(sorted(free_symbols, key=lambda x: x.name)[0])
    sorted_args = sorted(args, key=lambda x: index_arg_by_chars(x))
    first_expr = sorted_args[0]
    try:
        assert len(first_expr.free_symbols) == 1
    except:
        return expr
    coeff = extract_coefficient(first_expr)
    if coeff == 1:
        return expr
    adjusted_expr = simplify(expr * 1 / coeff)
    return adjusted_expr
    

class EquationKillerV2:
    solve_eqs = True  # whether to solve the equation in the intermediate process
    sym_simplify = True  # whether to apply symbol substitution simplification
    accurate_mode = True  # whether to use accurate mode
    solve_rank_deficient_eqs = False  # whether to solve rank deficient equations
    use_cache = False  # whether to use cache to store solved target equations
    cache_eqs = None  # <dict>, {tuple(str(eqs),): [(sym_str, value)]}
    cache_target = None  # <dict>, {tuple(str(eqs),): value}
    cache_equivalence = dict()  # <dict>, {sym: [(expr, [premise_id])]}
    cache_expanded_exprs = set() # <set>, {expr}
    cache_possible_values = dict()

    @staticmethod
    def empty_cache():
        EquationKillerV2.cache_eqs = dict()
        EquationKillerV2.cache_target = dict()  # <dict>, {tuple(str(eqs),): value}
        EquationKillerV2.cache_equivalence = dict()  # <dict>, {sym: [(expr, [premise_id])]}
        EquationKillerV2.cache_expanded_exprs = set() # <set>, {expr}
        EquationKillerV2.cache_possible_values = dict() # <dict>, {expr: possible equivalent values}

    @staticmethod
    def get_minimum_target_equations(target_expr, eqs):
        """
        Return minimum target equations. Called by function <EquationKillerV2.solve_target>.
        :param target_expr: Target Expression.
        :param eqs: Existing Equations.
        :return target_sym: Target symbols.
        :return mini_eqs: minimum equations rank by solving difficulty.
        :return n_m: number of equations and syms.
        """
        target_sym = symbols("t_s")
        eqs = [target_sym - target_expr] + eqs

        sym_to_eqs = {}  # dict, sym: [equation]
        for eq in eqs:
            for sym in eq.free_symbols:
                if sym in sym_to_eqs:
                    sym_to_eqs[sym].append(eq)
                else:
                    sym_to_eqs[sym] = [eq]

        mini_eqs = [eqs[0]]  # mini equations
        mini_syms = eqs[0].free_symbols  # sym of mini equations
        n_m = [(len(mini_eqs), len(mini_syms))]  # number of equations and variable

        related_eqs = []  # related eqs waiting to add
        for sym in mini_syms:
            for r_eq in sym_to_eqs[sym]:
                related_eqs.append(r_eq)
        related_eqs = list(set(related_eqs) - set(mini_eqs))

        while len(related_eqs) > 0:
            added_eq_id = 0
            added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
            added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)
            for i in range(1, len(related_eqs)):
                if len(related_eqs[i].free_symbols - mini_syms) < added_eq_n1:
                    added_eq_id = i
                    added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
                    added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)
                elif len(related_eqs[i].free_symbols - mini_syms) == added_eq_n1:
                    if len(related_eqs[i].free_symbols) > added_eq_n2:
                        added_eq_id = i
                        added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
                        added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)

            added_eq = related_eqs[added_eq_id]
            mini_eqs.append(added_eq)
            mini_syms |= added_eq.free_symbols
            n_m.append((len(mini_eqs), len(mini_syms)))

            for sym in added_eq.free_symbols:
                for r_eq in sym_to_eqs[sym]:
                    related_eqs.append(r_eq)
            related_eqs = list(set(related_eqs) - set(mini_eqs))

        return target_sym, mini_eqs, n_m

    @staticmethod
    def get_minimum_group_equations(eqs):
        """
        Return minimum group equations. Called by function <EquationKillerV2.solve_equations>.
        :param eqs: Equations.
        :return mini_eqs_list: minimum equations lists rank by solving difficulty.
        :return n_m: number of equations and syms.
        """

        sym_to_eqs = {}  # dict, sym: [equation]
        for eq in eqs:
            for sym in eq.free_symbols:
                if sym in sym_to_eqs:
                    sym_to_eqs[sym].append(eq)
                else:
                    sym_to_eqs[sym] = [eq]

        mini_eqs_lists = []  # mini equations
        n_m = []  # number of equations and variable

        added_eqs = set()
        for eq in eqs:
            if eq in added_eqs:
                continue
            added_eqs.add(eq)

            mini_eqs = [eq]  # mini equations
            mini_syms = eq.free_symbols  # sym of mini equations

            related_eqs = []  # related eqs waiting to add
            for sym in mini_syms:
                for r_eq in sym_to_eqs[sym]:
                    related_eqs.append(r_eq)
            related_eqs = list(set(related_eqs) - set(mini_eqs))

            if len(related_eqs) == 0:
                mini_eqs_lists.append(mini_eqs)
                n_m.append((len(mini_eqs), len(mini_syms)))  # add mini equations
                continue

            while True:
                added_eq_id = 0
                added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
                added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)
                for i in range(1, len(related_eqs)):
                    if len(related_eqs[i].free_symbols - mini_syms) < added_eq_n1:
                        added_eq_id = i
                        added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
                        added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)
                    elif len(related_eqs[i].free_symbols - mini_syms) == added_eq_n1:
                        if len(related_eqs[i].free_symbols) > added_eq_n2:
                            added_eq_id = i
                            added_eq_n1 = len(related_eqs[added_eq_id].free_symbols - mini_syms)
                            added_eq_n2 = len(related_eqs[added_eq_id].free_symbols)

                added_eq = related_eqs[added_eq_id]
                mini_eqs.append(added_eq)
                mini_syms |= added_eq.free_symbols
                added_eqs.add(added_eq)

                for sym in added_eq.free_symbols:
                    for r_eq in sym_to_eqs[sym]:
                        related_eqs.append(r_eq)
                related_eqs = list(set(related_eqs) - set(mini_eqs))

                if len(related_eqs) == 0:
                    mini_eqs_lists.append(mini_eqs)
                    n_m.append((len(mini_eqs), len(mini_syms)))  # add mini equations
                    break

        return mini_eqs_lists, n_m

    @staticmethod
    def get_minimum_syms(target_eqs, eqs):
        """
        Return minimum equation's syms. Called by function <Searcher.get_theorem_selection>.
        :param target_eqs: <list>, target Equations.
        :param eqs: <list>, existing Equations.
        :return syms: <set>, set of minimum equation's syms.
        """
        sym_to_eqs = {}  # dict, sym: [equation]
        for eq in target_eqs + eqs:
            for sym in eq.free_symbols:
                if sym in sym_to_eqs:
                    sym_to_eqs[sym].append(eq)
                else:
                    sym_to_eqs[sym] = [eq]

        mini_eqs = set(target_eqs)
        mini_syms = set()
        for eq in mini_eqs:
            mini_syms |= eq.free_symbols

        while True:
            new_sym = set()
            for sym in mini_syms:
                for eq in sym_to_eqs[sym]:
                    mini_eqs.add(eq)
                    new_sym |= eq.free_symbols
            new_sym = new_sym - mini_syms
            if len(new_sym) == 0:
                break
            mini_syms |= new_sym

        return mini_syms

    @staticmethod
    # @func_set_timeout(2)
    def simplification_value_replace(problem):
        """
        Simplify equations by replacing sym with known value.
        :param problem: Instance of class <Problem>.
        """
        update = True
        added_conditions = []
        while update:
            update = False
            remove_lists = set()  # equation to be deleted
            add_lists = []  # equation to be added

            for eq in problem.condition.simplified_equation:  # solve eq that only one sym unsolved
                if len(eq.free_symbols) != 1:
                    continue

                target_sym = list(eq.free_symbols)[0]
                try:
                    result = EquationKillerV2.solve(eq)  # solve equations
                except FunctionTimedOut:
                    msg = "Timeout when solve equation: {}".format(eq)
                    warnings.warn(msg)
                else:
                    if target_sym in result:
                        problem.set_value_of_sym(target_sym, result[target_sym],
                                                 tuple(problem.condition.simplified_equation[eq]))
                        remove_lists |= {eq}

            for eq in problem.condition.simplified_equation:  # value replace
                if eq in remove_lists:
                    continue
                # if len(eq.free_symbols) != 1:
                #     continue
                raw_eq = eq
                simplified = False
                added_premise = []
                for sym in eq.free_symbols:
                    if problem.condition.value_of_sym[sym] is None:
                        continue
                    simplified = True  # replace sym with value when the value known
                    eq = eq.subs(sym, problem.condition.value_of_sym[sym])
                    added_premise.append(problem.condition.get_id_by_predicate_and_item(
                        "Equation", sym - problem.condition.value_of_sym[sym]))
                    remove_lists |= {raw_eq}

                if not simplified:
                    continue

                if len(eq.free_symbols) == 0:  # no need to add new simplified equation when it's all sym known
                    continue
                else:  # add new simplified equation
                    update = True
                    premise = problem.condition.simplified_equation[raw_eq] + added_premise
                    add_lists.append((eq, premise))

            for remove_eq in remove_lists:  # remove useless equation
                problem.condition.simplified_equation.pop(remove_eq)
                EquationKillerV2.cache_expanded_exprs.add(remove_eq)

            for add_eq, premise in add_lists:  # remove useless equation
                update = problem.add('Equation', add_eq, premise, ('solve_eq', 'simplify_value_replace', None))
                if update:
                    premise_id = problem.condition.get_id_by_predicate_and_item("Equation", add_eq)
                    problem.condition.simplified_equation[add_eq] = [premise_id]
                    # EquationKillerV2.cache_expanded_exprs.add(add_eq)
                # add_eq = adjust_first_variable_coefficient(add_eq)
                # added_conditions.append(('Equation', add_eq, premise, ('solve_eq', None, None)))
        
        # for para in added_conditions:
        #     problem.add(*para)
        return

    @staticmethod
    # @func_set_timeout(2)
    def simplification_sym_replace(equations, target_sym):
        """ High level simplify based on symbol replacement."""
        update = True
        while update:
            update = False
            for i in range(len(equations)):
                eq = equations[i]

                if target_sym in eq.free_symbols or \
                        len(eq.free_symbols) != 2 or \
                        len(eq.atoms()) > 5:  # too many atoms, no need to replace
                    continue

                try:
                    result = EquationKillerV2.solve(eq, keep_sym=True)  # solve sym
                except FunctionTimedOut:
                    msg = "Timeout when solve equations: {}".format(equations)
                    warnings.warn(msg)
                    continue

                if len(result) == 0:  # no solved result
                    continue

                sym = list(result.keys())[0]
                has_float = False
                for atom in result[sym].atoms():
                    if isinstance(atom, Float):
                        has_float = True
                        break
                if has_float:  # float has precision error
                    continue

                for j in range(len(equations)):  # replace sym with solved sym_expr
                    if sym in equations[j].free_symbols:
                        equations[j] = equations[j].subs(sym, result[sym])
                        update = True

        for i in range(len(equations))[::-1]:  # remove 0
            if len(equations[i].free_symbols) == 0:
                equations.pop(i)

    @staticmethod
    # @func_set_timeout(2)
    def solve(equations, target_sym=None, keep_sym=False):
        if type(equations) in [tuple, list] and len(equations) > 6: 
            return {}
        if isinstance(equations, Expr):
            # if eq like: sin(x) + y = 0
            if variable_in_function(equations) and len(equations.free_symbols) >= 2:
                return {}
            if len(equations.free_symbols) >= 3:
                return {}

        try:
            if target_sym is not None:
                solved = solve(equations, target_sym, dict=True)
            else:
                solved = solve(equations, dict=True)

            if len(solved) == 0:  # no result solved
                return {}
        except Exception as e:  # exception
            msg = "Exception <{}> occur when solve {}".format(e, equations)
            warnings.warn(msg)
            return {}
        else:  # has result
            if keep_sym:  # keep sym result
                if isinstance(solved, list):
                    return solved[0]
                return solved

            if isinstance(solved, list):
                update = True
                while update and len(solved) > 1:  # choose min but > 0, when has multi result
                    update = False
                    for sym in solved[0]:
                        if sym not in solved[1]:
                            solved.pop(1)
                            update = True
                            break
                        if len(solved[0][sym].free_symbols) != 0 and len(solved[1][sym].free_symbols) == 0:
                            solved.pop(0)
                            update = True
                            break
                        if len(solved[0][sym].free_symbols) == 0 and len(solved[1][sym].free_symbols) != 0:
                            solved.pop(1)
                            update = True
                            break
                        if len(solved[0][sym].free_symbols) != 0:
                            continue
                        if solved[0][sym] == solved[1][sym]:
                            continue
                        if float(solved[0][sym]) < 0 < float(solved[1][sym]):
                            solved.pop(0)
                            update = True
                            break
                        if float(solved[0][sym]) > 0 > float(solved[1][sym]):
                            solved.pop(1)
                            update = True
                            break

                    if update:
                        continue

                    for sym in solved[0]:
                        if len(solved[0][sym].free_symbols) != 0:
                            continue
                        if solved[0][sym] == solved[1][sym]:
                            continue
                        if abs(float(solved[0][sym])) > abs(float(solved[1][sym])):
                            solved.pop(0)
                            update = True
                        else:
                            solved.pop(1)
                            update = True
                        break

                solved = solved[0]

            real_results = {}  # real_number
            for sym in solved:  # filter out real number solution
                if len(solved[sym].free_symbols) == 0:
                    # real_results[sym] = number_round(solved[sym])
                    real_results[sym] = solved[sym]
            return real_results
        
    @staticmethod
    def build_equivalence(problem, use_cache=False):
        """
        Build equivalent expressions / symbols for each symbol 
        Update problem.condition.equivalence_of_sym
        """
        # add equivalent exprs according to existed conditions / equations
        
        if len(EquationKillerV2.cache_equivalence) == 0 or not use_cache:
            temp_dict = copy.deepcopy(problem.condition.equivalence_of_sym)
            eqs_to_remove = set()
            add_paras = []
            for sym in problem.condition.value_of_sym:
                equivalent_expr_list = []
                for eq, premise_id in problem.condition.simplified_equation.items():
                    if sym not in eq.free_symbols:
                        continue
                    if 'll_ce**2' in str(eq): # debug
                        a = 1
                    if not check_expr(eq):
                        continue
                    # solve from a-b=0: -> a=b; a^2-b^2=0: -> a=b
                    result = EquationKillerV2.solve(eq, target_sym=sym, keep_sym=True)
                    if sym in result:
                        equivalent_expr_list.append((result[sym], premise_id))
                        new_expr = adjust_first_variable_coefficient(sym - result[sym])
                        # if constant item > 360, do not add
                        if abs(new_expr.as_coefficients_dict().get(1, 0)) <= 360:
                            add_paras.append(('Equation', new_expr, premise_id, ('solve_eq', 'build_equivalence', None)))
                        eqs_to_remove.add(eq)
                # update `problem.condition.equivalence_of_sym`
                for expand_v, expand_premise in equivalent_expr_list:
                    if expand_v != sym and expand_v not in [item[0] for item in temp_dict[sym]]:
                        temp_dict[sym].append((expand_v, expand_premise))

            for para in add_paras:
                problem.add(*para)
                # EquationKillerV2.cache_expanded_exprs.add(para[1])
            
            # delete used eqs in `problem.condition.simplified_equation`
            problem.condition.equivalence_of_sym = temp_dict
            for eq in list(eqs_to_remove):
                problem.condition.simplified_equation.pop(eq)
                EquationKillerV2.cache_expanded_exprs.add(eq)
            
            # expand equivalent exprs of one symbol according to expr substitute
            # a: [b, c, ..]
            # b: [a, v1, v2, ..]
            # -> a: [b, v1, v2, c, ..]
            # -> b: [a, v1, v2, c, ..]
            temp_dict = copy.deepcopy(problem.condition.equivalence_of_sym)
            for sym, equivalent_exprs in problem.condition.equivalence_of_sym.items():
                equivalent_exprs_expanded = []
                # existed_values = [item[0] for item in equivalent_exprs]
                for expr, premise_ids in equivalent_exprs:
                    # all possible values for `expr` through substitute equivalent value
                    expand_res = EquationKillerV2.get_all_equivalent_expr(
                        expr, premise_ids, problem.condition.equivalence_of_sym
                    )
                    equivalent_exprs_expanded += expand_res
                
                for expand_v, expand_premise in equivalent_exprs_expanded:
                    if expand_v != sym and expand_v not in [item[0] for item in temp_dict[sym]]:
                        # add in `condition.items`, but not in `condition.simplified_equation`
                        new_expr = adjust_first_variable_coefficient(sym - expand_v)
                        if abs(new_expr.as_coefficients_dict().get(1, 0)) <= 360:
                            if type(new_expr.as_coefficients_dict().get(1, 0)) in [int, Integer, Pow]:
                                problem.add('Equation', new_expr, expand_premise, 
                                        ('solve_eq', 'build_equivalence', None))
                                # add in `problem.condition.equivalence_of_sym` with new premise_id, not expand_premise
                                new_premise_id = [problem.condition.get_id_by_predicate_and_item("Equation", new_expr)]
                                temp_dict[sym].append((expand_v, new_premise_id))

            # simplify equivalence_of_sym
            for sym in temp_dict:
                equivalent_exprs = copy.deepcopy(temp_dict[sym])
                equivalent_exprs_to_remove = []
                for expr, premise_ids in equivalent_exprs:
                    # all symbols from expr are known as number value
                    if all([problem.condition.value_of_sym[sym_i] is not None for sym_i in expr.free_symbols]):
                        equivalent_exprs_to_remove.append((expr, premise_ids))
                        if problem.condition.value_of_sym[sym] is None:
                            value_dict = dict(zip(expr.free_symbols, [problem.condition.value_of_sym[sym_i] for sym_i in expr.free_symbols]))
                            _premise_ids = copy.deepcopy(premise_ids)
                            for sym_i, value_i in value_dict.items():
                                _premise_ids.append(problem.condition.get_id_by_predicate_and_item('Equation', sym_i - value_i))

                            problem.set_value_of_sym(sym, expr.subs(value_dict), _premise_ids)
                
                for expr, premise_ids in equivalent_exprs_to_remove:
                    equivalent_exprs.remove((expr, premise_ids))
                
                temp_dict[sym] = equivalent_exprs

            problem.condition.equivalence_of_sym = temp_dict
            EquationKillerV2.cache_equivalence = temp_dict
        
        else:
            return
                    
    @staticmethod
    def get_all_equivalent_expr(expr, premise_ids, values_dict, 
                               stop_when_solved_num=False, 
                               use_cache=False):
        # substitute all combinations of equivalent value for symbols in expr, return all possible values
        if 'll_ce**2' in str(expr): # debug
             a = 1
        # if expr has >= 2 perimeter or area symbol
        if not check_expr_perimeter_area(expr):
            return []
        
        variables = list(expr.free_symbols)
        value_combinations = list(product(
            *[values_dict[var] if len(values_dict[var]) > 0 else [(var, [])]
              for var in variables]
        ))
        # value_combinations = list(product(*[values_dict[var] for var in variables]))
        substitute_res = []
        if not use_cache or expr not in EquationKillerV2.cache_possible_values:
            for values_group in value_combinations:
                values = [item[0] for item in values_group]
                premises = [item for sub_lst in values_group for item in sub_lst[1]]
                premises = sorted(set(premises))
                if len(premises) == 0:
                    continue
                # get all symbols from values list
                values_symbols = set()
                for value in values:
                    values_symbols.update(value.free_symbols)
                # if do not have simplification
                if len(expr.free_symbols) < len(values_symbols):
                    continue
                subs_dict = dict(zip(variables, values))
                new_expr = expr.subs(subs_dict)
                
                # if check_expr(new_expr, min_symbol_num=1, max_items_num=2):
                new_premises = sorted(set(premise_ids + premises))
                substitute_res.append((new_expr, new_premises))
                if len(new_premises) > 3:
                    a = 1
                if stop_when_solved_num and len(new_expr.free_symbols) == 0:
                    break

            EquationKillerV2.cache_possible_values[expr] = substitute_res

            
        return EquationKillerV2.cache_possible_values[expr]
    
    @staticmethod
    def simplify_through_equivalence(problem):
        """
        simplify eqs in `problem.condition.simplified_equation` through_equivalence
        simplified: the number of unknown variables decreases
        """
        eqs_to_remove = set()
        eqs_to_add = list()
        # 遍历所有简化的方程
        for eq_expr in list(problem.condition.simplified_equation):
            # 获取所有可能的等价替换结果
            update = False
            expand_res = EquationKillerV2.get_all_equivalent_expr(
                eq_expr, 
                problem.condition.simplified_equation[eq_expr],
                problem.condition.equivalence_of_sym,
                stop_when_solved_num=True
            )
            # EquationKillerV2.solve()
            if 'll_ce**2' in str(eq_expr): # debug
                a = 1
            if 0 in [item[0] for item in expand_res]:
                eqs_to_remove.add(eq_expr)
                continue
            symbols_num_ori = len(eq_expr.free_symbols)
            # 将新的等价表达式添加到condition中
            for expand_expr, expand_premise in expand_res:
                if expand_expr != eq_expr and len(expand_expr.free_symbols) < symbols_num_ori and len(expand_expr.free_symbols) > 0:
                    if check_expr(expand_expr):
                        # 先简化expand_expr，将类似a/2 + b/2 变为 a + b
                        new_expr = adjust_first_variable_coefficient(expand_expr)
                        eqs_to_add.append((new_expr, expand_premise))
                        # problem.add('Equation', new_expr, expand_premise, ('solve_eq', None, None))

                        update = True

            if update:
                eqs_to_remove.add(eq_expr)

        for eq in list(eqs_to_remove):
            problem.condition.simplified_equation.pop(eq)
            EquationKillerV2.cache_expanded_exprs.add(eq)
        for eq, premise in eqs_to_add:
            update = problem.add('Equation', eq, premise, ('solve_eq', 'simplify_through_equivalence', None))
            if update:
                premise_id = problem.condition.get_id_by_predicate_and_item("Equation", eq)
                problem.condition.simplified_equation[eq] = [premise_id]
                # EquationKillerV2.cache_expanded_exprs.add(eq)
        return

    @staticmethod
    def solve_through_equivalence(problem):
        # if solved sym = value, the equivalent expr of the sym equals to this value
        for sym in problem.condition.equivalence_of_sym:
            value = problem.condition.value_of_sym[sym]
            if value is None:
                continue
            
            sym_premise = problem.condition.get_id_by_predicate_and_item('Equation', sym - value)
            for expr, premise_ids in problem.condition.equivalence_of_sym[sym]:
                new_expr = expr - value
                if len(new_expr.free_symbols) == 1:
                    sym = new_expr.free_symbols.pop()
                    try:
                        results = EquationKillerV2.solve(new_expr, sym)  # solve equations
                    except FunctionTimedOut:
                        msg = "Timeout when solve equations: {}".format(new_expr)
                        warnings.warn(msg)
                    else:
                        if sym in results:
                            problem.set_value_of_sym(sym, results[sym], premise_ids + [sym_premise])
        
        # equivalent exprs of each symbol are equal
        for sym, equivalent_exprs in problem.condition.equivalence_of_sym.items():
            value_combinations = list(combinations(equivalent_exprs, 2))
            for item_1, item_2 in value_combinations:
                expr_1, premise_1 = item_1
                expr_2, premise_2 = item_2
                new_expr = expr_1 - expr_2 
                if len(new_expr.free_symbols) == 1:
                    sym = new_expr.free_symbols.pop()
                    try:
                        results = EquationKillerV2.solve(new_expr, sym)  # solve equations
                    except FunctionTimedOut:
                        msg = "Timeout when solve equations: {}".format(new_expr)
                        warnings.warn(msg)
                    else:
                        if sym in results:
                            problem.set_value_of_sym(sym, results[sym], premise_1 + premise_2)
        
        # equivalent exprs is equal to value of symbol (if have number value)
        for sym, equivalent_exprs in problem.condition.equivalence_of_sym.items():
            value = problem.condition.value_of_sym[sym]
            if value is not None:
                for expr, premise_ids in equivalent_exprs:
                    new_expr = expr - value
                    if len(new_expr.free_symbols) == 1:
                        sym = new_expr.free_symbols.pop()
                        try:
                            results = EquationKillerV2.solve(new_expr, sym)  # solve equations
                        except FunctionTimedOut:
                            msg = "Timeout when solve equations: {}".format(new_expr)
                            warnings.warn(msg)
                        else:
                            if sym in results:
                                problem.set_value_of_sym(sym, results[sym], premise_ids)
        

        return

    @staticmethod
    def solve_equations(problem):
        """
        Solve equations in problem.condition.equations.
        :param problem: Instance of class <Problem>.
        """
        if not EquationKillerV2.solve_eqs or problem.condition.eq_solved:
            return

        try:
            EquationKillerV2.simplification_value_replace(problem)  # simplify equations before solving
        except FunctionTimedOut:
            msg = "Timeout when simplify equations by value replace."
            warnings.warn(msg)
            
        try:
            EquationKillerV2.build_equivalence(problem)
        except FunctionTimedOut:
            msg = "Timeout when building equivalent expressions for symbol."
            warnings.warn(msg)

        try:
            EquationKillerV2.simplify_through_equivalence(problem)
        except FunctionTimedOut:
            msg = "Timeout when simplify equations through equivalence."
            warnings.warn(msg)
            
        try:
            EquationKillerV2.solve_through_equivalence(problem)
        except FunctionTimedOut:
            msg = "Timeout when solving equations through equivalence expressions."
            warnings.warn(msg)

        # delete too complex equations first
        too_complex_eqs = set()
        for eq in list(problem.condition.simplified_equation):
            if variable_in_function(eq) and len(eq.free_symbols) >= 3:
                too_complex_eqs.add(eq)

        for eq in list(too_complex_eqs):
            problem.condition.simplified_equation.pop(eq)

        # first try to solve single equation one by one
        single_solved = False
        eqs_to_remove = set()
        for eq in list(problem.condition.simplified_equation):
            try:
                results = EquationKillerV2.solve(eq)  # solve equations
            except FunctionTimedOut:
                msg = "Timeout when solve equations: {}".format(eq)
                warnings.warn(msg)
            else:
                for sym in results:
                    if problem.condition.value_of_sym[sym] is None:
                        single_solved = True
                        problem.set_value_of_sym(sym, results[sym], problem.condition.simplified_equation[eq])
                        eqs_to_remove.add(eq)

        if single_solved:
            try:
                EquationKillerV2.simplification_value_replace(problem)
            except FunctionTimedOut:
                msg = "Timeout when simplify equations through equivalence."
                warnings.warn(msg)

        # solve equations by groups
        mini_eqs_lists, n_m = EquationKillerV2.get_minimum_group_equations(  # get mini equations
            list(problem.condition.simplified_equation)
        )

        for i in range(len(mini_eqs_lists)):
            if not EquationKillerV2.solve_rank_deficient_eqs and n_m[i][0] < n_m[i][1]:
                continue

            eqs_for_cache = None
            if EquationKillerV2.use_cache:
                eqs_for_cache = []
                premise = []
                str_to_sym = {}
                for eq in mini_eqs_lists[i]:
                    eqs_for_cache.append(str(eq))
                    premise += problem.condition.simplified_equation[eq]
                    for sym in eq.free_symbols:
                        str_to_sym[str(sym)] = sym
                eqs_for_cache = tuple(sorted(eqs_for_cache))

                if eqs_for_cache in EquationKillerV2.cache_eqs:
                    for sym_str, value in EquationKillerV2.cache_eqs[eqs_for_cache]:
                        problem.set_value_of_sym(str_to_sym[sym_str], value, premise)
                    continue
                EquationKillerV2.cache_eqs[eqs_for_cache] = []

            solved = False
            solved_results = None
            mini_eqs = None

            try:
                results = EquationKillerV2.solve(mini_eqs_lists[i])  # solve equations
            except FunctionTimedOut:
                msg = "Timeout when solve equations: {}".format(mini_eqs_lists[i])
                warnings.warn(msg)
            else:
                for sym in results:
                    if problem.condition.value_of_sym[sym] is None:
                        solved = True
                        solved_results = results
                        mini_eqs = mini_eqs_lists[i]
                        break

            if not solved:
                continue

            if EquationKillerV2.accurate_mode:
                for sym in solved_results:
                    if problem.condition.value_of_sym[sym] is not None:
                        continue
                    sym_mini_eqs = copy.copy(mini_eqs)
                    for removed_eq in copy.copy(sym_mini_eqs):
                        try_eqs = copy.copy(sym_mini_eqs)
                        try_eqs.remove(removed_eq)
                        try:
                            results = EquationKillerV2.solve(try_eqs, sym)  # solve equations
                        except FunctionTimedOut:
                            msg = "Timeout when solve equations: {}".format(try_eqs)
                            warnings.warn(msg)
                        else:
                            if sym in results:
                                sym_mini_eqs.remove(removed_eq)

                    premise = []
                    for eq in sym_mini_eqs:
                        premise += problem.condition.simplified_equation[eq]
                    problem.set_value_of_sym(sym, solved_results[sym], premise)

                    if EquationKillerV2.use_cache:
                        EquationKillerV2.cache_eqs[eqs_for_cache].append((str(sym), solved_results[sym]))

            else:
                premise = []
                for eq in mini_eqs:
                    premise += problem.condition.simplified_equation[eq]

                for sym in solved_results:
                    problem.set_value_of_sym(sym, solved_results[sym], premise)

                if EquationKillerV2.use_cache:
                    for sym in solved_results:
                        EquationKillerV2.cache_eqs[eqs_for_cache].append((str(sym), solved_results[sym]))

    @staticmethod
    def solve_target(target_expr, problem):
        """
        Solve target_expr in the constraint of problem's equation.
        :param problem: Instance of class <Problem>.
        :param target_expr: symbol expression.
        """
        if target_expr is None:
            return None, []

        if target_expr in problem.condition.get_items_by_predicate("Equation"):  # no need to solve
            return 0, [problem.condition.get_id_by_predicate_and_item("Equation", target_expr)]
        if -target_expr in problem.condition.get_items_by_predicate("Equation"):
            return 0, [problem.condition.get_id_by_predicate_and_item("Equation", -target_expr)]

        

        # start_time = time.time()
        try:
            EquationKillerV2.build_equivalence(problem, use_cache=True)
        except FunctionTimedOut:
            msg = "Timeout when building equivalent expressions for symbol."
            warnings.warn(msg)
        # print(f"build_equivalence: {time.time() - start_time}")
     
        # solve only using value replacement
        premise = []
        for sym in target_expr.free_symbols:  
            if problem.condition.value_of_sym[sym] is not None:
                target_expr = target_expr.subs(sym, problem.condition.value_of_sym[sym])
                premise.append(problem.condition.get_id_by_predicate_and_item(
                    "Equation", sym - problem.condition.value_of_sym[sym]))
        if len(target_expr.free_symbols) == 0:
            return target_expr, premise

        # if total degree >= 2: return None directly for saving time
        if total_degree(target_expr) >= 2:
            return None, []

        # solve through equivalent exprs replacement
        expand_res = EquationKillerV2.get_all_equivalent_expr(
            target_expr, 
            [],
            problem.condition.equivalence_of_sym,
            stop_when_solved_num=True,
            use_cache=True
        )

        solved_sym = [sym for sym in problem.condition.value_of_sym
                      if problem.condition.value_of_sym[sym] is not None]
        _premise_ids = []
        for new_expr, premise_ids in expand_res:
            # if returns number, return directly
            if len(new_expr.free_symbols) == 0:
                return new_expr, premise_ids
            # if all symbols of new_expr are solved to num
            if all([s in solved_sym for s in list(new_expr.free_symbols)]):
                for sym in new_expr.free_symbols:
                    new_expr = new_expr.subs(sym, problem.condition.value_of_sym[sym])
                    _premise_ids.append(problem.condition.get_id_by_predicate_and_item(
                        "Equation", sym - problem.condition.value_of_sym[sym]))
                if len(new_expr.free_symbols) == 0:
                    return new_expr, premise_ids + _premise_ids
                
        return None, []
    


class GeometryPredicateLogicExecutorV2:

    @staticmethod
    def run(gpl, problem, letters=None):
        """
        Run reason step by step.
        :param gpl: <dict>, (products, logic_constraints, algebra_constraints, conclusions), geometric predicate logic.
        :param problem: instance of class <Problem>.
        :param letters: preset letters for para selection.
        :return results: <list> of <tuple>, [(letters, premises, conclusions)].
        """
        r = GeometryPredicateLogicExecutorV2.run_logic(gpl, problem, letters)
        r = GeometryPredicateLogicExecutorV2.run_algebra(r, gpl, problem)
        return GeometryPredicateLogicExecutorV2.make_conclusion(r, gpl, problem)

    @staticmethod
    def run_logic(gpl, problem, letters=None):
        """
        Run 'products', 'logic_constraints' of GPL.
        :param gpl: <dict>, (products, logic_constraints, algebra_constraints, conclusions), geometric predicate logic.
        :param problem: instance of class <Problem>.
        :param letters: preset letters for para selection.
        :return r: triplet, (r_ids, r_items, r_vars).
        """
        products = gpl["products"]
        logic_constraints = gpl["logic_constraints"]

        r_ids, r_items = problem.condition.get_ids_and_items_by_predicate_and_variable(products[0][0], products[0][1])
        r_vars = products[0][1]
        for i in range(len(r_items)):  # delete duplicated vars and corresponding item
            r_items[i] = list(r_items[i])
        r_vars = list(r_vars)
        deleted_vars_index = []  # deleted vars index
        for i in range(len(r_vars)):
            if r_vars[i] in r_vars[0:i]:
                deleted_vars_index.append(i)
        for index in deleted_vars_index[::-1]:  # delete
            r_vars.pop(index)
            for i in range(len(r_items)):
                r_items[i].pop(index)

        for i in range(1, len(products)):
            r_ids, r_items, r_vars = GeometryPredicateLogicExecutorV2.product(
                (r_ids, r_items, r_vars), products[i], problem)

        if letters is not None:  # select result according to letters
            for i in range(len(r_ids))[::-1]:
                selected = True
                for v in letters:
                    if r_items[i][r_vars.index(v)] != letters[v]:
                        selected = False
                        break
                if not selected:
                    r_items.pop(i)
                    r_ids.pop(i)

        for i in range(len(logic_constraints)):
            r_ids, r_items, r_vars = GeometryPredicateLogicExecutorV2.constraint_logic(
                (r_ids, r_items, r_vars), logic_constraints[i], problem)

        if letters is not None:  # select result according to letters
            for i in range(len(r_ids))[::-1]:
                selected = True
                for v in letters:
                    if r_items[i][r_vars.index(v)] != letters[v]:
                        selected = False
                        break
                if not selected:
                    r_items.pop(i)
                    r_ids.pop(i)

        return r_ids, r_items, r_vars

    @staticmethod
    def run_algebra(r, gpl, problem):
        """
        Run 'algebra_constraints' of GPL.
        :param r: triplet, (r_ids, r_items, r_vars).
        :param gpl: <dict>, (products, logic_constraints, algebra_constraints, conclusions), geometric predicate logic.
        :param problem: instance of class <Problem>.
        :return results: <list> of <tuple>, [(letters, premises, conclusions)].
        """
        algebra_constraints = gpl["algebra_constraints"]
        r_ids, r_items, r_vars = r
        if len(r_ids) == 0:
            return [], [], r_vars

        for i in range(len(algebra_constraints)):
            r_ids, r_items, r_vars = GeometryPredicateLogicExecutorV2.constraint_algebra(
                (r_ids, r_items, r_vars), algebra_constraints[i], problem)

        return r_ids, r_items, r_vars

    @staticmethod
    def make_conclusion(r, gpl, problem):
        """
        Make conclusion according given reasoned points sets 'r' and GDL 'conclusions'.
        :param r: triplet, (r_ids, r_items, r_vars).
        :param gpl: <dict>, (products, logic_constraints, algebra_constraints, conclusions), geometric predicate logic.
        :param problem: instance of class <Problem>.
        :return results: <list> of <tuple>, [(letters, premises, conclusions)].
        """
        if len(r[0]) == 0:
            return []
        conclusions = gpl["conclusions"]
        results = []
        r_ids, r_items, r_vars = r
        for i in range(len(r_ids)):
            letters = {}
            for j in range(len(r_vars)):
                letters[r_vars[j]] = r_items[i][j]
            conclusion = []

            for predicate, item in conclusions:
                if predicate == "Equal":  # algebra conclusion
                    eq = get_equation_from_tree(problem, item, True, letters)
                    conclusion.append(("Equation", eq))
                else:  # logic conclusion
                    item = tuple(letters[i] for i in item)
                    conclusion.append((predicate, item))
            results.append((letters, r_ids[i], conclusion))

        return results

    @staticmethod
    def product(r1, r2_logic, problem):
        """
        Constrained Cartesian product.
        :param r1: triplet, (r1_ids, r1_items, r1_vars).
        :param r2_logic: geo predicate logic, such as ['Collinear', ['a', 'b', 'c']].
        :param problem: instance of class <Problem>.
        :return r: triplet, (r_ids, r_items, r_vars), reasoning result.
        >> product(([(1,), (2,)], [('A', 'B'), ('C', 'D')], ['a', 'b']),
                   ['Line', ['b', 'c']],
                   problem)
        ([(1, 3), (2, 4)], [('A', 'B', 'C'), ('C', 'D', 'E')], ['a', 'b', 'c'])
        """
        r1_ids, r1_items, r1_vars = r1
        if len(r1_ids) == 0:
            return [], [], r1_vars
        r2_ids, r2_items = problem.condition.get_ids_and_items_by_predicate_and_variable(r2_logic[0], r2_logic[1])
        r2_vars = r2_logic[1]

        inter = list(set(r1_vars) & set(r2_vars))  # intersection
        for i in range(len(inter)):
            inter[i] = (r1_vars.index(inter[i]), r2_vars.index(inter[i]))  # change to index

        difference = list(set(r2_vars) - set(r1_vars))  # difference
        for i in range(len(difference)):
            difference[i] = r2_vars.index(difference[i])  # change to index

        r_ids = []  # result
        r_items = []
        r_vars = list(r1_vars)
        for dif in difference:  # add r2 vars
            r_vars.append(r2_vars[dif])
        r_vars = tuple(r_vars)

        for i in range(len(r1_items)):
            r1_data = r1_items[i]
            for j in range(len(r2_items)):
                r2_data = r2_items[j]
                passed = True
                for r1_i, r2_i in inter:
                    if r1_data[r1_i] != r2_data[r2_i]:  # the corresponding points are inconsistent.
                        passed = False
                        break
                if passed:
                    item = list(r1_data)
                    for dif in difference:
                        item.append(r2_data[dif])
                    r_items.append(tuple(item))
                    r_ids.append(tuple(set(list(r1_ids[i]) + list(r2_ids[j]))))
        return r_ids, r_items, r_vars

    @staticmethod
    def constraint_logic(r1, r2_logic, problem):
        """
        Logic constraint.
        :param r1: triplet, (r1_ids, r1_items, r1_vars).
        :param r2_logic: geo predicate logic, such as ['Collinear', ['a', 'b', 'c']].
        :param problem: instance of class <Problem>.
        :return r: triplet, (r_ids, r_items, r_vars), reasoning result.
        >> problem.conditions['Line'].get_item_by_id  # supposed
        {3: ('B', 'C')}
        >> constraint_logic(([(1,), (2,)], [('A', 'B', 'C'), ('C', 'D', 'E')], ['a', 'b', 'c']),
                            ['Line', ['b', 'c']],
                            problem)
        ([(1, 3)], [('A', 'B', 'C')], ['a', 'b', 'c'])
        >> constraint_logic(([(1,), (2,)], [('A', 'B', 'C'), ('C', 'D', 'E')], ['a', 'b', 'c']),
                            ['~Line', ['b', 'c']],
                            problem)
        ([(2,)], [('C', 'D', 'E')], ['a', 'b', 'c'])
        """
        r1_ids, r1_items, r1_vars = r1
        if len(r1_ids) == 0:
            return [], [], r1_vars
        oppose = False  # indicate '&' or '&~'
        if "~" in r2_logic[0]:
            r2_logic = list(r2_logic)
            r2_logic[0] = r2_logic[0].replace("~", "")
            r2_logic = tuple(r2_logic)
            oppose = True
        index = [r1_vars.index(v) for v in r2_logic[1]]
        r_ids = []
        r_items = []

        if not oppose:  # &
            for i in range(len(r1_items)):
                r2_item = tuple(r1_items[i][j] for j in index)
                if r2_item in problem.condition.get_items_by_predicate(r2_logic[0]):
                    r2_id = problem.condition.get_id_by_predicate_and_item(r2_logic[0], r2_item)
                    r_ids.append(tuple(set(list(r1_ids[i]) + [r2_id])))
                    r_items.append(r1_items[i])
        else:  # &~
            for i in range(len(r1_items)):
                r2_item = tuple(r1_items[i][j] for j in index)
                if r2_item not in problem.condition.get_items_by_predicate(r2_logic[0]):
                    r_ids.append(r1_ids[i])
                    r_items.append(r1_items[i])
        return r_ids, r_items, r1_vars

    @staticmethod
    def constraint_algebra(r1, r2_algebra, problem):
        """
        Algebra constraint.
        :param r1: triplet, (r1_ids, r1_items, r1_vars).
        :param r2_algebra: geo predicate logic, such as ['Equal', [['Length', ['a', 'b']], 5]].
        :param problem: instance of class <Problem>.
        :return r: triplet, (r_ids, r_items, r_vars), reasoning result.
        >> problem.conditions['Equation'].get_value_of_sym  # supposed
        {ll_ab: 1}
        >> problem.conditions['Equation'].get_item_by_id  # supposed
        {3: ll_ab - 1}
        >> constraint_algebra(([(1,), (2,)], [('A', 'B', 'C'), ('C', 'D', 'E')], ['a', 'b', 'c']),
                              ['Equal', [['Length', ['a', 'b']], 1]],
                              problem)
        ([(1, 3)], [('A', 'B', 'C')], ['a', 'b', 'c'])
        >> constraint_algebra(([(1,), (2,)], [('A', 'B', 'C'), ('C', 'D', 'E')], ['a', 'b', 'c']),
                              ['~Equal', [['Length', ['a', 'b']], 1]],
                              problem)
        ([(2,)], [('C', 'D', 'E')], ['a', 'b', 'c'])
        """
        r1_ids, r1_items, r1_vars = r1
        if len(r1_ids) == 0:
            return [], [], r1_vars
        oppose = False  # indicate '&' or '&~'
        if "~" in r2_algebra[0]:
            r2_algebra = list(r2_algebra)
            r2_algebra[0] = r2_algebra[0].replace("~", "")
            r2_algebra = tuple(r2_algebra)
            oppose = True
        r_ids = []
        r_items = []
        
        cached_eq_result = {}
        if not oppose:  # &
            for i in range(len(r1_items)):
                letters = {}
                for j in range(len(r1_vars)):
                    letters[r1_vars[j]] = r1_items[i][j]
                eq = get_equation_from_tree(problem, r2_algebra[1], True, letters)
                if eq in cached_eq_result:
                    result, premise = cached_eq_result[eq]
                else:
                    try:
                        result, premise = EquationKillerV2.solve_target(eq, problem)
                        cached_eq_result[eq] = (result, premise)
                    except FunctionTimedOut:
                        msg = "Timeout when solve target: {}".format(str(eq))
                        pass

                if result is not None and rough_equal(result, 0):  # meet constraints
                    r_id = tuple(set(premise + list(r1_ids[i])))
                    r_ids.append(r_id)
                    r_items.append(r1_items[i])
        else:  # &~
            for i in range(len(r1_items)):
                letters = {}
                for j in range(len(r1_vars)):
                    letters[r1_vars[j]] = r1_items[i][j]
                eq = get_equation_from_tree(problem, r2_algebra[1], True, letters)
                try:
                    result, premise = EquationKillerV2.solve_target(eq, problem)
                except FunctionTimedOut:
                    msg = "Timeout when solve target: {}".format(str(eq))
                    warnings.warn(msg)
                else:
                    if result is None or not rough_equal(result, 0):  # meet constraints
                        r_id = tuple(set(premise + list(r1_ids[i])))
                        r_ids.append(r_id)
                        r_items.append(r1_items[i])

        return r_ids, r_items, r1_vars
