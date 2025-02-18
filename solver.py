import random
import time
from collections import defaultdict
from copy import copy, deepcopy
from functools import wraps

import sympy as sp
from func_timeout import FunctionTimedOut, func_timeout
from sympy import Eq, cos, rad, simplify, sin, solve, sqrt, tan
from tqdm import tqdm

from formalgeo.core import EquationKiller as EqKiller
from formalgeo.core import GeometryPredicateLogicExecutor as GPLExecutor
from formalgeo.parse import (parse_predicate_gdl, parse_problem_cdl,
                             parse_theorem_gdl)
from formalgeo.problem import Problem
from formalgeo.tools import debug_print, get_used_pid_and_theorem
from inter_gps_solver.extended_definition import ExtendedDefinition
from inter_gps_solver.logic_parser import LogicParser
from inter_gps_solver.logic_solver import LogicSolver
from utils.symbolic import degree_of_expr, formalgeo_to_intergps, parse_clause


def make_timing_decorator(debug_attr='debug'):
    def decorator(func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            start_time = time.time()
            result = func(instance, *args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Check instance's debug attribute and print only if True
            if getattr(instance, debug_attr, False):
                print(f"Function [{func.__name__}] took {elapsed_time:.6f} seconds to execute.")
            
            return result
        return wrapper
    return decorator

def get_p2t_map_fw(t_info, parsed_theorem_GDL):
    """
    Predicate-theorem mapping hash table for Search Accelerating.
    :param t_info: <dict>, {t_name: (category_id, usage_count)}, user customization.
    :param parsed_theorem_GDL: parsed theorem GDL.
    """
    p2t_map_fw = {}
    for t_name in t_info:
        if t_name == 'angle_addition': 
            a = 1
        if t_info[t_name][1] == 0 or t_info[t_name][0] == 3:  # skip no used and diff t
            continue
        for t_branch in parsed_theorem_GDL[t_name]["body"]:
            theorem_unit = parsed_theorem_GDL[t_name]["body"][t_branch]
            premises = list(theorem_unit["products"])
            premises += list(theorem_unit["logic_constraints"])
            premises += list(theorem_unit["attr_in_algebra_constraints"])
            for predicate, p_vars in premises:
                if predicate[0] == "~":  # skip oppose
                    continue
                if predicate not in p2t_map_fw:
                    p2t_map_fw[predicate] = [(t_name, t_branch, p_vars)]
                elif (t_name, t_branch, p_vars) not in p2t_map_fw[predicate]:
                    p2t_map_fw[predicate].append((t_name, t_branch, p_vars))
    return p2t_map_fw


class FormalGeoSolver:
    def __init__(self, predicate_GDL, theorem_GDL, strategy, max_depth, beam_size, t_info, t_freq_info, p_pos=None, debug=False):
        """
        Initialize Forward Searcher.
        :param predicate_GDL: predicate GDL.
        :param theorem_GDL: theorem GDL.
        :param strategy: <str>, "dfs", "bfs", "rs", "bs".
        :param max_depth: max search depth.
        :param beam_size: beam search size.
        :param t_info: <dict>, {t_name: (category_id, usage_count)}, user customization.
        :param debug: <bool>, set True when need print process information.
        """
        self.parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, self.parsed_predicate_GDL)
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.strategy = strategy
        self.p_pos = p_pos # pass in pos to judge counter clockwise
        self.debug = debug
        self.p2t_map = get_p2t_map_fw(t_info, self.parsed_theorem_GDL)
        self.t_freq_info = t_freq_info
        
        self.problem = None
        self.stack = None
        self.last_step = None
        self.step_size = None
        self.node_count = None  # {depth: node_count}

        self.problem_p_paras = None  # Perimeter
        self.problem_a_paras = None  # Area
        
    @make_timing_decorator('debug')
    def init_search(self, problem_CDL):
        """Initial problem by problem_CDL and build root Node."""
        EqKiller.use_cache = True  # use cache to speed up solving
        EqKiller.cache_eqs = {}
        EqKiller.cache_target = {}

        timing = time.time()  # timing

        self.problem = Problem(p_pos=self.p_pos)  # init problem
        self.problem.load_problem_by_fl(
            self.parsed_predicate_GDL, self.parsed_theorem_GDL, parse_problem_cdl(problem_CDL))
        EqKiller.solve_equations(self.problem)
        self.problem.step("init_problem", 0)

        self.stack = []
        self.last_step = 0
        self.step_size = 0
        self.node_count = {1: 1}

        self.problem_p_paras = set()  # Perimeter
        self.problem_a_paras = set()  # Area
        for sym in self.problem.condition.attr_of_sym:
            predicate, paras = self.problem.condition.attr_of_sym[sym]
            if predicate.startswith("Perimeter"):
                for para in paras:
                    self.problem_p_paras.add(para)
            elif predicate.startswith("Area"):
                for para in paras:
                    self.problem_a_paras.add(para)

        debug_print(self.debug, "(pid={}, strategy={}, timing={:.4f}s) Initialize and start forward search...".format(
            problem_CDL["problem_id"], self.strategy, time.time() - timing))

        timing = time.time()
        selections = self.get_theorem_selection()
        self.add_selections([], selections)
        debug_print(self.debug, "(timing={:.4f}s) Expand {} child node.".
                    format(time.time() - timing, len(selections)))
        
    def solve_special_angles(self):
        right_triangles = self.problem.condition.get_items_by_predicate('RightTriangle')
        
        for tri in right_triangles:
            premise_right_triangle = self.problem.condition.get_id_by_predicate_and_item('RightTriangle', tri)
            angle_chars = [tri, 
                           (tri[1], tri[2], tri[0]), 
                           (tri[2], tri[0], tri[1])]
            angle_syms = [self.problem.get_sym_of_attr('MeasureOfAngle', angle) for angle in angle_chars]
            angle_values = [self.problem.condition.value_of_sym[sym] for sym in angle_syms]
            
            special_angle = None
            if 60 in angle_values and 90 in angle_values:
                special_angle = 60
            elif 30 in angle_values and 90 in angle_values:
                special_angle = 30
            elif 45 in angle_values and 90 in angle_values:
                special_angle = 45
            
            if special_angle is not None:
                angle_sp = angle_chars[angle_values.index(special_angle)]
                angle_90 = angle_chars[angle_values.index(90)]
                angle_sp_sym = angle_syms[angle_values.index(special_angle)]
                angle_90_sym = angle_syms[angle_values.index(90)]
                # l2 = cos(a) * l1
                # l3 = sin(a) * l1
                # l3 = tan(a) * l2
                l1 = (angle_sp[0], angle_sp[1])
                l2 = (angle_sp[1], angle_sp[2])
                l3 = (angle_sp[2], angle_sp[0])
                if angle_sp[0] == angle_90[1]:
                    l1, l2 = l2, l1
                l1_sym = self.problem.get_sym_of_attr('LengthOfLine', l1)
                l2_sym = self.problem.get_sym_of_attr('LengthOfLine', l2)
                l3_sym = self.problem.get_sym_of_attr('LengthOfLine', l3)
                
                expr_1 = l2_sym - cos(rad(special_angle)) * l1_sym
                expr_2 = l3_sym - sin(rad(special_angle)) * l1_sym
                expr_3 = l3_sym - tan(rad(special_angle)) * l2_sym
                premise_expr_1 = angle_sp_sym - special_angle
                # premise_expr_2 = angle_90_sym - 90
                premise = [premise_right_triangle]
                
                for i, c in enumerate(self.problem.condition.items):
                    if c[0] == 'Equation':
                        if c[1] == premise_expr_1:
                            premise.append(i)
                
                last_step = len(self.problem.condition.items)
                self.problem.add('Equation', expr_1, premise, ('cos_of_angle', None, None))
                self.problem.add('Equation', expr_2, premise, ('sin_of_angle', None, None))
                self.problem.add('Equation', expr_3, premise, ('tan_of_angle', None, None))
                
                # add
                if len(self.problem.condition.items) - last_step > 0:
                    new_condition = deepcopy(self.problem.condition.items[last_step:])
                    # find position in search tree
                    pos = None
                    for k, v in self.leveled_condition.items():
                        ids = list(v.keys())
                        if premise[0] in ids or premise[1] in ids:
                            pos = k
                            break
                    if pos is None:
                        pos = list(self.leveled_condition.keys())[-1]
                        depth = len(pos)
                        new_pos = pos[:-1] + (pos[-1] + 1,)
                    else:
                        depth = len(pos)
                        if depth not in self.node_count:
                            self.node_count[depth] = 1
                        new_pos = tuple(list(pos) + [self.node_count[depth]])
                    self.node_count[depth] += 1
                    self.leveled_condition[new_pos] = {}
                    for i in range(len(new_condition)):
                        self.leveled_condition[new_pos][last_step+i] = new_condition[i]
                
                    self.problem.step(('solve_eq', None, None), 0)
        return
        
    
    def solve_equations(self):
        # solve equation, substitute value of syms first
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        value_of_sym = {
            k: v for k, v in 
            self.problem.condition.value_of_sym.items()
            if v is not None
        }
        for expr in eq_exprs:
            solved = False 
            eq_idx = eq_exprs[expr][0]
            premise = [eq_idx]
            syms_in_expr = list(expr.free_symbols)
            expr_new = copy(expr)
            
            for sym, value in value_of_sym.items():
                if sym in syms_in_expr and value != None:
                    premise_expr = sym - value
                    for i, c in enumerate(self.problem.condition.items):
                        if c[0] == 'Equation' and c[1] == premise_expr:
                            premise.append(i)
                    expr_new = expr_new.subs(sym, value)
            
            last_step = len(self.problem.condition.items)
            
            # can solve to value directly
            if len(expr_new.free_symbols) == 1 and len(expr.free_symbols) > 1:
                sym = list(expr_new.free_symbols)[0]
                eq = Eq(expr_new, 0)
                results = solve(eq, sym)

                if self.problem.condition.value_of_sym[sym] is None and len(results) != 0:
                    solved = True
                    solved_results = results[0]
                    self.problem.set_value_of_sym(sym, solved_results, premise)     

            # can get new numerical relation though substitute
            elif len(expr_new.free_symbols) == 2 and len(expr.free_symbols) > 2:
                solved = True
                self.problem.add('Equation', expr_new, premise, ('solve_eq', None, None))
                # remove the past equation
                self.problem.condition.simplified_equation.pop(expr)
                # add condition idx of new equation
                self.problem.condition.simplified_equation[expr_new] = [len(self.problem.condition.items) - 1]
                    
            if not solved:
                continue
            
            # add condition in `leveled_condition`
            if len(self.problem.condition.items) - last_step > 0:
                new_condition = deepcopy(self.problem.condition.items[last_step:])
                # find position in search tree
                pos = list(self.leveled_condition.keys())[-1]
                for k, v in self.leveled_condition.items():
                    if eq_idx in list(v.keys()):
                        pos = k
                # get new depth and pos
                depth = len(pos)
                if depth not in self.node_count:
                    self.node_count[depth] = 1
                new_pos = tuple(list(pos) + [self.node_count[depth]])
                self.node_count[depth] += 1
                self.leveled_condition[new_pos] = {}
                for i in range(len(new_condition)):
                    self.leveled_condition[new_pos][last_step+i] = new_condition[i]
                    
                self.problem.step(('solve_eq', None, None), 0)

        return 
    
    def solve_equivalence_relation(self):
        def set2key(set_input):
            res = sorted(set_input, key=lambda x: str(x))
            return tuple(res)
        
        def contains_sqrt(expr):
            return any(isinstance(term, sp.Pow) and term.exp == sp.S.Half for term in sp.preorder_traversal(expr))
        
        def is_equal_expr(expr):
            f1 = len(expr.free_symbols) == 2
            f2 = all(abs(v) == 1 for v in expr.as_coefficients_dict().values()) 
            f3 = not contains_sqrt(expr)
            return f1 and f2 and f3
        
        # expand equivalence relation in equation groups
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        # split expr like: ll_ab - ll_cd
        expr_list = [
            expr for expr in eq_exprs if is_equal_expr(expr)
        ]

        eq_sym_groups = [] # list of set (symbols)
        eq_groups_to_ids = {} # dict: tuple (symbols) -> id set (premise)

        # find equivalent class (symbols group)
        for expr in expr_list:
            var_a, var_b = list(expr.free_symbols)
            class_a, class_b = None, None
            for sym_group in eq_sym_groups:
                if var_a in sym_group:
                    class_a = sym_group
                if var_b in sym_group:
                    class_b = sym_group
            
            # merge symbols group, remove origin and add new
            if class_a is not None and class_b is not None:
                if class_a != class_b:
                    eq_sym_groups.remove(class_a)
                    eq_sym_groups.remove(class_b)
                    eq_sym_groups.append(class_a.union(class_b))
                    id_1 = eq_groups_to_ids.pop(set2key(class_a))
                    id_2 = eq_groups_to_ids.pop(set2key(class_b))
                    # premise id from class a, class b, new expr
                    eq_groups_to_ids[set2key(class_a.union(class_b))] = set(list(id_1) + list(id_2) + eq_exprs[expr])
            elif class_a is not None:
                id_1 = eq_groups_to_ids.pop(set2key(class_a))
                class_a.add(var_b)
                eq_groups_to_ids[set2key(class_a)] = set(list(id_1) + eq_exprs[expr])

            elif class_b is not None:
                id_1 = eq_groups_to_ids.pop(set2key(class_b))
                class_b.add(var_a)
                eq_groups_to_ids[set2key(class_b)] = set(list(id_1) + eq_exprs[expr])
            else:
                eq_sym_groups.append({var_a, var_b})
                eq_groups_to_ids[set2key({var_a, var_b})] = set(eq_exprs[expr])
        
        # get new expr
        new_expr_list = []
        premise_list = []
        for sym_group_set in eq_sym_groups:
            sym_group = list(sym_group_set)
            for i in range(len(sym_group) - 1):
                for j in range(i + 1, len(sym_group)):
                    new_expr_1 = sym_group[i] - sym_group[j]
                    new_expr_2 = sym_group[j] - sym_group[i]
                    if new_expr_1 not in expr_list and new_expr_2 not in expr_list:
                        new_expr_list.append(new_expr_1)
                        premise_list.append(list(eq_groups_to_ids[set2key(sym_group)]))
        
        for expr, premise in zip(new_expr_list, premise_list):
            # add condition in `items`
            last_step = len(self.problem.condition.items)
            self.problem.add('Equation', expr, premise, ('solve_eq', None, None))
            new_condition = deepcopy(self.problem.condition.items[last_step:])
            
            if len(new_condition) > 0:
                # add condition in `leveled_condition`
                pos = list(self.leveled_condition.keys())[-1]
                for k, v in self.leveled_condition.items():
                    if premise[0] in list(v.keys()):
                        pos = k
                        
                depth = len(pos)
                if depth not in self.node_count:
                    self.node_count[depth] = 1
                new_pos = tuple(list(pos) + [self.node_count[depth]])
                self.node_count[depth] += 1
                self.leveled_condition[new_pos] = {}
                for i in range(len(new_condition)):
                    self.leveled_condition[new_pos][last_step+i] = new_condition[i]
                
            self.problem.step(('solve_eq', None, None), 0)
            
        return 


    @make_timing_decorator('debug')
    def solve_equation_groups(self):
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        eqs = [simplify(Eq(e, 0)) for e in eq_exprs]
        update = False
        
        for i, expr_i in enumerate(eq_exprs):
            if degree_of_expr(expr_i) >= 2:
                continue
            if len(expr_i.free_symbols) != 2:
                continue
            # assert expr_i has 2 free symbols
            related_exprs = [
                expr for expr in list(eq_exprs.keys())[i+1:]
                if len(expr_i.free_symbols & expr.free_symbols) == 2
            ]
            last_step = len(self.problem.condition.items)
            
            for expr_j in related_exprs:
                premise = eq_exprs[expr_i] + eq_exprs[expr_j]

                # 1. if expr_j is linear, use sympy `solve`
                # 2. if expr_j's degree == 2, substitute to get new expr
                if degree_of_expr(expr_j) == 1:
                    results = solve((expr_i, expr_j), expr_i.free_symbols, dict=True)
                elif degree_of_expr(expr_j) == 2:
                    sym_1, sym_2 = list(expr_i.free_symbols)
                    expr_i_result = solve(expr_i, sym_1, dict=True)
                    if isinstance(expr_i_result, list):
                        expr_i_result = expr_i_result[0]
                    expr_new = expr_j.subs(expr_i_result)
                    results = solve(expr_new, sym_2, dict=True)
                else:
                    continue
                
                if len(results) == 0:
                    continue
                if isinstance(results, list):
                    results = results[0]
                    
                # add conditions
                for k, v in results.items():
                    expr_new = k - v
                    # check if already in equations
                    if simplify(Eq(expr_new, 0)) in eqs:
                        continue
                    self.problem.add('Equation', expr_new, premise, ('solve_eq', None, None))
                    self.problem.condition.simplified_equation[expr_new] = [len(self.problem.condition.items) - 1]
                    
            
            # add condition
            if len(self.problem.condition.items) - last_step > 0:
                new_condition = deepcopy(self.problem.condition.items[last_step:])
                # find position in search tree
                pos = list(self.leveled_condition.keys())[-1]
                for k, v in self.leveled_condition.items():
                    if max(premise) in list(v.keys()):
                        pos = k
                # get new depth and pos
                depth = len(pos)
                if depth not in self.node_count:
                    self.node_count[depth] = 1
                new_pos = tuple(list(pos) + [self.node_count[depth]])
                self.node_count[depth] += 1
                self.leveled_condition[new_pos] = {}
                for i in range(len(new_condition)):
                    self.leveled_condition[new_pos][last_step+i] = new_condition[i]
                update = True
        if update:
            self.problem.step(('solve_eq', None, None), 0)
        return 
    
    def search(self):
        """
        Search problem and return search result.
        :return solved: <bool>, indicate whether problem solved or not.
        :return seqs: <list> of <str>, solved theorem sequences.
        self.problem.condition.items
        """
        self.leveled_condition = {}
        cur_depth = 0
        while len(self.stack) > 0:
            beam_count = len(self.stack)
            if cur_depth == 0 or cur_depth == 1:
                beam_count = 200
            elif cur_depth == 2:
                beam_count = 100
            elif cur_depth == 3:
                beam_count = 50
            else:
                beam_count = self.beam_size
                    
            if len(self.stack) > beam_count:  
                stack = []
                for i in random.sample(range(len(self.stack)), beam_count):
                    stack.append(self.stack[i])
                self.stack = stack
                # beam_count = self.beam_size
            else:
                beam_count = len(self.stack)

            for i in range(beam_count):
                pos, selection = self.stack.pop(0)
                self.step_size += 1
                # debug_print(self.debug, "\n(pos={}, node_count={}) Current node.".format(pos, self.node_count))
                timing = time.time()
                
                # save leveled condition, new added
                # last_step = len(self.problem.condition.items)
                solved = self.apply_and_check_goal(selection, pos)
                
                # debug_print(self.debug, "(solved={}, timing={:.4f}s) Apply selection and check goal.".format(solved, time.time() - timing))
                if solved is None:  # not update, close search branch
                    continue
                if solved:  # solved, return result
                    _, seqs = get_used_pid_and_theorem(self.problem)
                    return True, seqs
                else:  # continue search
                    if len(pos) >= self.max_depth:
                        continue
                    timing = time.time()
                    selections = self.get_theorem_selection(
                        # sample_num_per_th=3
                    )
                    self.add_selections(pos, selections)
                    debug_print(self.debug, "(timing={:.4f}s) Expand {} child node.".
                                format(time.time() - timing, len(selections)))
            
            
            self.problem.check_goal()
            solved = self.problem.goal.solved
            if solved:  # solved, return result
                _, seqs = get_used_pid_and_theorem(self.problem)
                return True, seqs
            
            self.solve_special_angles()
            self.solve_equations()
            self.solve_equation_groups()
            self.solve_equivalence_relation()
            selections = self.get_theorem_selection()
            self.add_selections(pos, selections)
            cur_depth += 1
        
        # EqKiller.solve_equations(self.problem)
        return False, None
    
    def get_theorem_selection(self, sample_num_per_th=None):
        """
        Return theorem selections according to <self.last_step>.
        :return selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        selections = []

        timing = time.time()
        related_pres = []  # new added predicates
        related_syms = []  # new added/updated equations
        for step in range(self.last_step, self.problem.condition.step_count):  # get related conditions
        # for step in range(0, self.problem.condition.step_count):
            for _id in self.problem.condition.ids_of_step[step]:
                if self.problem.condition.items[_id][0] == "Equation":
                    for sym in self.problem.condition.items[_id][1].free_symbols:
                        if sym in related_syms:
                            continue
                        related_syms.append(sym)
                else:
                    if self.problem.condition.items[_id][0] not in self.p2t_map:
                        continue
                    item = self.problem.condition.items[_id][1]
                    for t_name, t_branch, p_vars in self.p2t_map[self.problem.condition.items[_id][0]]:
                        if len(p_vars) != len(item):
                            continue
                        letters = {}
                        for i in range(len(p_vars)):
                            letters[p_vars[i]] = item[i]
                        related_pre = (t_name, t_branch, letters)
                        if related_pre not in related_pres:
                            related_pres.append(related_pre)
        # debug_print(self.debug, "(timing={:.4f}s) Get Related.".format(time.time() - timing))
        # debug_print(self.debug, "Related predicates: {}.".format(related_pres))
        # debug_print(self.debug, "Related syms: {}.".format(related_syms))

        timing = time.time()
        logic_selections = self.try_theorem_logic(
            related_pres, 
            sample_num_per_th
        )
        # debug_print(self.debug, "(timing={:.4f}s) Get {} logic-related selections: {}.".format(time.time() - timing, len(logic_selections), logic_selections))
        timing = time.time()
        algebra_selections = self.try_theorem_algebra(
            related_syms, 
            sample_num_per_th
        )
        # algebra_selections = []
        # debug_print(self.debug, "(timing={:.4f}s) Get {} algebra-related selections: {}.".format(time.time() - timing, len(algebra_selections), algebra_selections))
        # debug_print(f'logic: {len(logic_selections)}')
        # debug_print(f'algeb: {len(algebra_selections)}')
        
        timing = time.time()
        added_selections = []
        for selection in logic_selections + algebra_selections:  # remove redundancy
            _, conclusions = selection
            s = []
            for conclusion in conclusions:
                predicate, item, _ = conclusion
                s.append((predicate, item))
            s = tuple(s)
            if s not in added_selections:
                added_selections.append(s)
                selections.append(selection)

        for i in range(len(selections))[::-1]:
            t_msg, conclusions = selections[i]
            t_name, t_branch, t_para = t_msg
            if "area" in t_name:
                if "ratio" in t_name:
                    para1 = t_para[0:int(len(t_para) / 2)]
                    para2 = t_para[int(len(t_para) / 2):]
                    if not (para1 in self.problem_a_paras and para2 in self.problem_a_paras):
                        selections.pop(i)
                else:
                    if t_para not in self.problem_a_paras:
                        selections.pop(i)
            elif "perimeter" in t_name:
                if "ratio" in t_name:
                    para1 = t_para[0:int(len(t_para) / 2)]
                    para2 = t_para[int(len(t_para) / 2):]
                    if not (para1 in self.problem_p_paras and para2 in self.problem_p_paras):
                        selections.pop(i)
                else:
                    if t_para not in self.problem_p_paras:
                        selections.pop(i)

            
        # debug_print(self.debug, "(timing={:.4f}s) Get {}  selections: {}.".format(time.time() - timing, len(selections), selections))

        selections = sorted(
            selections, 
            key=lambda item: self.t_freq_info[item[0][0]],
            reverse=True
        )
        return selections
    
    def pre_select_theorems_logic(self, selections, sample_num_per_th=None):
        if sample_num_per_th == None:
            sample_num_per_th = self.beam_size * 2
        selection_dict = {}
        for item in selections:
            if item[0] not in selection_dict: 
                selection_dict[item[0]] = [item]
            else:
                selection_dict[item[0]].append(item)
        
        reserved_num = 0
        step_1_selections = []
        remained_selections = []
        for k, v in selection_dict.items():
            reserved_num += min(sample_num_per_th, len(v))
            step_1_selections += random.sample(
                v, min(sample_num_per_th, len(v)))
            remained_selections += [s for s in v if s not in step_1_selections]
        
        remaining_sample_size = min(len(selections), 2000) - reserved_num
        step_2_selections = sorted(
            remained_selections, 
            key=lambda item: self.t_freq_info[item[0]],
            reverse=True
        )[:remaining_sample_size]
        
        pre_selections = sorted(
            step_1_selections + step_2_selections, 
            key=lambda item: self.t_freq_info[item[0]],
            reverse=True
        )
        return pre_selections
    
    def pre_select_theorems_algebra(self, selections, sample_num_per_th=None):

        if sample_num_per_th == None:
            sample_num_per_th = self.beam_size
            
        selection_dict = {}
        for item in selections:
            if item[0] not in selection_dict: 
                selection_dict[item[0]] = [item]
            else:
                selection_dict[item[0]].append(item)
        
        pre_selections = []
        for k, v in selection_dict.items():
            if 'similar' in k or 'congruent' in k:
                continue
            else:
                pre_selections += random.sample(
                    v, min(sample_num_per_th, len(v)))
            # if len(v) > sample_num_per_th:
            #     pre_selections += random.sample(v, sample_num_per_th)
            # else:
            #     pre_selections += v
            
        pre_selections = sorted(
            pre_selections, 
            key=lambda item: self.t_freq_info[item[0]],
            reverse=True
        )
        # frequency = [self.t_freq_info[item[0]] for item in pre_selections]
        return pre_selections

    @make_timing_decorator('debug')
    def try_theorem_logic(self, related_pres, sample_num_per_th=None):
        """
        Try a theorem and return can-added conclusions.
        :param related_pres: <list>, list of tuple('t_name', 't_branch', letters).
        :return selections: <list> of ((t_name, t_branch, t_para, t_timing), ((predicate, item, premise))).
        """
        if sample_num_per_th is None:
            sample_num_per_th = self.beam_size * 2
        selections = []
        # related_pres_ = self.pre_select_theorems_logic(
        #     related_pres, 
        #     sample_num_per_th=sample_num_per_th
        # )
        debug_theorems = [
            'collinear_judgement_angle'
        ]
        for t_name, t_branch, t_letters in related_pres:
            if 'similar_arc' in t_name:
                continue
            if t_name in debug_theorems:
                a = 1
            gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]
            results = GPLExecutor.run(gpl, self.problem, t_letters)  # get gpl reasoned result
            for letters, premise, conclusion in results:
                # fix bugs in `congruent_arc_judgment`
                if 'congruent_arc_judgment' in t_name:
                    arc_1_set = set([letters['a'], letters['b']])
                    arc_2_set = set([letters['c'], letters['d']])
                    if len(arc_1_set & arc_2_set) == 2: 
                        continue
                    
                if t_name in debug_theorems:
                    a = 1
                
                t_para = tuple([letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]])

                premise = tuple(premise)
                conclusions = []
                for predicate, item in conclusion:  # add conclusion
                    if self.problem.check(predicate, item, premise, t_name):
                        if predicate != "Equation":
                            item = tuple(item)
                        conclusions.append((predicate, item, premise))

                if len(conclusions) > 0:
                    selections.append(((t_name, t_branch, t_para), tuple(conclusions)))
                    # if len(selections) > self.beam_size:
                    #     return selections
        # sele_to_remove = []
        # for sele in selections:
        #     if sele[0][0] == 'round_angle':
        #         sele_to_remove.append(sele)
        # for sele in sele_to_remove:
        #     selections.remove(sele)
            
        return selections

    @make_timing_decorator('debug')
    def try_theorem_algebra(self, related_syms, sample_num_per_th=None):
        """
        Try a theorem and return can-added conclusions.
        :param related_syms: <list>, related syms.
        :return selections: <list> of ((t_name, t_branch, t_para, t_timing), ((predicate, item, premise))).
        """
        if sample_num_per_th is None:
            sample_num_per_th = self.beam_size 
        paras_of_attrs = {}  # <dict>, {attr: [para]}
        for sym in related_syms:
            attr, paras = self.problem.condition.attr_of_sym[sym]
            if attr not in self.p2t_map:
                continue

            if attr not in paras_of_attrs:
                paras_of_attrs[attr] = []

            for para in paras:
                if para in paras_of_attrs[attr]:
                    continue
                paras_of_attrs[attr].append(para)

        selections = []
        for related_attr in paras_of_attrs:
            related_paras = set(paras_of_attrs[related_attr])
            related_theorems = self.pre_select_theorems_algebra(self.p2t_map[related_attr], sample_num_per_th)
            # if related_attr == 'MeasureOfAngle':
            #     a = 1
            for t_name, t_branch, p_vars in related_theorems:
                gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]  # run gdl
                for related_para in related_paras:
                    letters = {}
                    for i in range(len(p_vars)):
                        letters[p_vars[i]] = related_para[i]
                    results = GPLExecutor.run(gpl, self.problem, letters)  # get gpl reasoned result
                    for letters, premise, conclusion in results:
                        theorem_para = tuple([letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]])
                        premise = tuple(premise)
                        conclusions = []
                        for predicate, item in conclusion:  # add conclusion
                            if self.problem.check(predicate, item, premise, t_name):
                                if predicate != "Equation":
                                    item = tuple(item)
                                conclusions.append((predicate, item, premise))

                        if len(conclusions) > 0:
                            selections.append(((t_name, t_branch, theorem_para), tuple(conclusions)))
                            # if len(selections) > self.beam_size:
                            #     return selections
        return selections

    def apply_and_check_goal(self, selection, pos):
        """
        Apply selection and check goal.
        :param selection: ((t_name, t_branch, t_para), ((predicate, item, premise))).
        :return solved: <bool> or None. Set None when not update
        """
        self.last_step = self.problem.condition.step_count
        update = False
        t_msg, conclusions = selection
        
        if 'square_property_cocircular' == t_msg[0]:
            a = 1
        last_idx = len(self.problem.condition.items)    
        for predicate, item, premise in conclusions:
            update = self.problem.add(predicate, item, premise, t_msg, skip_check=True) or update

        if not update:  # close current branch if applied theorem no new condition
            return None
        
        EqKiller.solve_equations(self.problem)  # solve eq & check_goal
    
        # add conditions in `leveled_condition`
        if len(self.problem.condition.items) - last_idx > 0:
            new_condition = deepcopy(self.problem.condition.items[last_idx:])
            self.leveled_condition[pos] = {}
            for i in range(len(new_condition)):
                self.leveled_condition[pos][last_idx+i] = new_condition[i]
                
        # use self-implemented methods to transfer into nature language more easily
        # self.solve_special_angles()
        # self.solve_equations()
        # self.solve_equation_groups()
        
        self.problem.check_goal()
        self.problem.step(t_msg, 0)

        return self.problem.goal.solved

    def add_selections(self, father_pos, selections):
        """
        Add selections to self.stack.
        :param father_pos: position of father branch.
        :param selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        pos = list(father_pos)
        depth = len(father_pos) + 1
        if depth not in self.node_count:
            self.node_count[depth] = 1

        for selection in selections:
            self.stack.append((tuple(pos + [self.node_count[depth]]), selection))
            self.node_count[depth] += 1



class Interactor:

    def __init__(self, predicate_GDL, theorem_GDL, t_info=None, debug=False):
        """
        Initialize Interactor.
        :param predicate_GDL: predicate GDL.
        :param theorem_GDL: theorem GDL.
        """
        self.parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, self.parsed_predicate_GDL)
        self.problem = None
        
        self.debug = debug
        self.last_step = 0
        self.p2t_map = get_p2t_map_fw(t_info, self.parsed_theorem_GDL)
        
        self.problem_p_paras = set()  # Perimeter
        self.problem_a_paras = set()  # Area
        self.ignore_ops = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle', 'Arc']

    def load_problem(self, problem_CDL):
        """Load problem through problem_CDL."""
        start_time = time.time()
        self.problem = Problem()
        self.problem.load_problem_by_fl(self.parsed_predicate_GDL,
                                        self.parsed_theorem_GDL,
                                        parse_problem_cdl(problem_CDL))  # load problem
        EqKiller.solve_equations(self.problem)  # Solve the equations after initialization
        self.problem.step("init_problem", time.time() - start_time)  # save applied theorem and update step

    def apply_theorem(self, t_name, t_branch=None, t_para=None):
        """
        Apply a theorem and return whether it is successfully applied.
        :param t_name: <str>.
        :param t_branch: <str>.
        :param t_para: tuple of <str>.
        :return update: <bool>, Whether the question condition updated or not.
        """
        # self.last_step = self.problem.condition.step_count
        
        if self.problem is None:
            e_msg = "Problem not loaded. Please run <load_problem> before run <apply_theorem>."
            raise Exception(e_msg)
        if t_name not in self.parsed_theorem_GDL:
            e_msg = "Theorem {} not defined in current GDL.".format(t_name)
            raise Exception(e_msg)
        if t_name.endswith("definition"):
            e_msg = "Theorem {} only used for backward reason.".format(t_name)
            raise Exception(e_msg)
        if t_para is not None and len(t_para) != len(self.parsed_theorem_GDL[t_name]["vars"]):
            e_msg = "Theorem <{}> para length error. Expected {} but got {}.".format(
                t_name, len(self.parsed_theorem_GDL[t_name]["vars"]), t_para)
            raise Exception(e_msg)
        if t_branch is not None and t_branch not in self.parsed_theorem_GDL[t_name]["body"]:
            e_msg = "Theorem <{}> branch error. Expected {} but got {}.".format(
                t_name, self.parsed_theorem_GDL[t_name]["body"].keys(), t_branch)
            raise Exception(e_msg)


        update = self.apply_theorem_by_name(t_name)

        if not update:
            w_msg = "Theorem <{},{},{}> not applied. Please check your theorem or prerequisite.".format(
                t_name, t_branch, t_para)

        return update

    def apply_theorem_by_name(self, t_name):
        """
        Apply a theorem with t_name.
        :param t_name: <str>.
        :return update: <bool>, Whether the problem condition updated or not.
        """
        update = False

        for branch in self.parsed_theorem_GDL[t_name]["body"]:
            timing = time.time()  # timing
            gpl = self.parsed_theorem_GDL[t_name]["body"][branch]

            conclusions = GPLExecutor.run(gpl, self.problem)  # get gpl reasoned result
            if len(conclusions) == 0:
                theorem = (t_name, branch, None)
                self.problem.step(theorem, time.time() - timing)
                continue
            avg_timing = (time.time() - timing) / len(conclusions)
            for letters, premise, conclusion in conclusions:
                t_para = [letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]]
                theorem = (t_name, branch, tuple(t_para))
                for predicate, item in conclusion:  # add conclusion
                    update = self.problem.add(predicate, item, premise, theorem) or update
                self.problem.step(theorem, avg_timing)

        timing = time.time()  # timing
        self.solve_special_angles()
        self.solve_equations()
        self.solve_equation_groups()
        self.solve_equivalence_relation()
        EqKiller.solve_equations(self.problem)
        self.problem.step("solve_equations", time.time() - timing)

        return update
    
    def solve_special_angles(self):
        right_triangles = self.problem.condition.get_items_by_predicate('RightTriangle')
        for tri in right_triangles:
            angle_chars = [tri, 
                           (tri[1], tri[2], tri[0]), 
                           (tri[2], tri[0], tri[1])]
            angle_syms = [self.problem.get_sym_of_attr('MeasureOfAngle', angle) for angle in angle_chars]
            angle_values = [self.problem.condition.value_of_sym[sym] for sym in angle_syms]
            
            special_angle = None
            if 60 in angle_values and 90 in angle_values:
                special_angle = 60
            elif 30 in angle_values and 90 in angle_values:
                special_angle = 30
            elif 45 in angle_values and 90 in angle_values:
                special_angle = 45
            
            if special_angle is not None:
                angle_sp = angle_chars[angle_values.index(special_angle)]
                angle_90 = angle_chars[angle_values.index(90)]
                angle_sp_sym = angle_syms[angle_values.index(special_angle)]
                angle_90_sym = angle_syms[angle_values.index(90)]
                # l2 = cos(a) * l1
                # l3 = sin(a) * l1
                # l3 = tan(a) * l2
                l1 = (angle_sp[0], angle_sp[1])
                l2 = (angle_sp[1], angle_sp[2])
                l3 = (angle_sp[2], angle_sp[0])
                if angle_sp[0] == angle_90[1]:
                    l1, l2 = l2, l1
                l1_sym = self.problem.get_sym_of_attr('LengthOfLine', l1)
                l2_sym = self.problem.get_sym_of_attr('LengthOfLine', l2)
                l3_sym = self.problem.get_sym_of_attr('LengthOfLine', l3)
                
                expr_1 = l2_sym - cos(rad(special_angle)) * l1_sym
                expr_2 = l3_sym - sin(rad(special_angle)) * l1_sym
                expr_3 = l3_sym - tan(rad(special_angle)) * l2_sym
                premise_expr_1 = angle_sp_sym - special_angle
                premise_expr_2 = angle_90_sym - 90
                premise = []
                for i, c in enumerate(self.problem.condition.items):
                    if c[0] == 'Equation':
                        if c[1] == premise_expr_1 or c[1] == premise_expr_2:
                            premise.append(i)
                assert len(premise) == 2
                
                last_step = len(self.problem.condition.items)
                self.problem.add('Equation', expr_1, premise, ('cos_of_angle', None, None))
                self.problem.add('Equation', expr_2, premise, ('sin_of_angle', None, None))
                self.problem.add('Equation', expr_3, premise, ('tan_of_angle', None, None))
                
                # add
                if len(self.problem.condition.items) - last_step > 0:
                    self.problem.step(('solve_eq', None, None), 0)
        return
        
    
    def solve_equations(self):
        # solve equation, substitute value of syms first
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        value_of_sym = {
            k: v for k, v in 
            self.problem.condition.value_of_sym.items()
            if v is not None
        }
        for expr in eq_exprs:
            solved = False 
            eq_idx = eq_exprs[expr][0]
            premise = [eq_idx]
            syms_in_expr = list(expr.free_symbols)
            expr_new = copy(expr)
            
            for sym, value in value_of_sym.items():
                if sym in syms_in_expr and value != None:
                    premise_expr = sym - value
                    for i, c in enumerate(self.problem.condition.items):
                        if c[0] == 'Equation' and c[1] == premise_expr:
                            premise.append(i)
                    expr_new = expr_new.subs(sym, value)
            
            last_step = len(self.problem.condition.items)
            
            # can solve to value directly
            if len(expr_new.free_symbols) == 1 and len(expr.free_symbols) > 1:
                sym = list(expr_new.free_symbols)[0]
                eq = Eq(expr_new, 0)
                results = solve(eq, sym)

                if self.problem.condition.value_of_sym[sym] is None and len(results) != 0:
                    solved = True
                    solved_results = results[0]
                    self.problem.set_value_of_sym(sym, solved_results, premise)     

            # can get new numerical relation though substitute
            elif len(expr_new.free_symbols) == 2 and len(expr.free_symbols) > 2:
                solved = True
                self.problem.add('Equation', expr_new, premise, ('solve_eq', None, None))
                # remove the past equation
                self.problem.condition.simplified_equation.pop(expr)
                # add condition idx of new equation
                self.problem.condition.simplified_equation[expr_new] = [len(self.problem.condition.items) - 1]
                    
            if not solved:
                continue
            
            # add condition in `leveled_condition`
            if len(self.problem.condition.items) - last_step > 0:  
                self.problem.step(('solve_eq', None, None), 0)

        return 
    
    def solve_equivalence_relation(self):
        def set2key(set_input):
            res = sorted(set_input, key=lambda x: str(x))
            return tuple(res)
        
        def contains_sqrt(expr):
            return any(isinstance(term, sp.Pow) and term.exp == sp.S.Half for term in sp.preorder_traversal(expr))
        
        def is_equal_expr(expr):
            f1 = len(expr.free_symbols) == 2
            f2 = all(abs(v) == 1 for v in expr.as_coefficients_dict().values()) 
            f3 = not contains_sqrt(expr)
            return f1 and f2 and f3
        
        # expand equivalence relation in equation groups
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        # split expr like: ll_ab - ll_cd
        expr_list = [
            expr for expr in eq_exprs if is_equal_expr(expr)
        ]

        eq_sym_groups = [] # list of set (symbols)
        eq_groups_to_ids = {} # dict: tuple (symbols) -> id set (premise)

        # find equivalent class (symbols group)
        for expr in expr_list:
            var_a, var_b = list(expr.free_symbols)
            class_a, class_b = None, None
            for sym_group in eq_sym_groups:
                if var_a in sym_group:
                    class_a = sym_group
                if var_b in sym_group:
                    class_b = sym_group
            
            # merge symbols group, remove origin and add new
            if class_a is not None and class_b is not None:
                if class_a != class_b:
                    eq_sym_groups.remove(class_a)
                    eq_sym_groups.remove(class_b)
                    eq_sym_groups.append(class_a.union(class_b))
                    id_1 = eq_groups_to_ids.pop(set2key(class_a))
                    id_2 = eq_groups_to_ids.pop(set2key(class_b))
                    # premise id from class a, class b, new expr
                    eq_groups_to_ids[set2key(class_a.union(class_b))] = set(list(id_1) + list(id_2) + eq_exprs[expr])
            elif class_a is not None:
                id_1 = eq_groups_to_ids.pop(set2key(class_a))
                class_a.add(var_b)
                eq_groups_to_ids[set2key(class_a)] = set(list(id_1) + eq_exprs[expr])

            elif class_b is not None:
                id_1 = eq_groups_to_ids.pop(set2key(class_b))
                class_b.add(var_a)
                eq_groups_to_ids[set2key(class_b)] = set(list(id_1) + eq_exprs[expr])
            else:
                eq_sym_groups.append({var_a, var_b})
                eq_groups_to_ids[set2key({var_a, var_b})] = set(eq_exprs[expr])
        
        # get new expr
        new_expr_list = []
        premise_list = []
        for sym_group_set in eq_sym_groups:
            sym_group = list(sym_group_set)
            for i in range(len(sym_group) - 1):
                for j in range(i + 1, len(sym_group)):
                    new_expr_1 = sym_group[i] - sym_group[j]
                    new_expr_2 = sym_group[j] - sym_group[i]
                    if new_expr_1 not in expr_list and new_expr_2 not in expr_list:
                        new_expr_list.append(new_expr_1)
                        premise_list.append(list(eq_groups_to_ids[set2key(sym_group)]))
        
        for expr, premise in zip(new_expr_list, premise_list):
            # add condition in `items`
            last_step = len(self.problem.condition.items)
            self.problem.add('Equation', expr, premise, ('solve_eq', None, None))
            new_condition = deepcopy(self.problem.condition.items[last_step:])
            
            if len(new_condition) > 0:
                self.problem.step(('solve_eq', None, None), 0)
            
        return 

    def solve_equation_groups(self):
        eq_exprs = deepcopy(self.problem.condition.simplified_equation)
        eqs = [simplify(Eq(e, 0)) for e in eq_exprs]
        update = False
        
        for i, expr_i in enumerate(eq_exprs):
            if degree_of_expr(expr_i) >= 2:
                continue
            if len(expr_i.free_symbols) != 2:
                continue
            # assert expr_i has 2 free symbols
            related_exprs = [
                expr for expr in list(eq_exprs.keys())[i+1:]
                if len(expr_i.free_symbols & expr.free_symbols) == 2
            ]
            last_step = len(self.problem.condition.items)
            
            for expr_j in related_exprs:
                premise = eq_exprs[expr_i] + eq_exprs[expr_j]

                # 1. if expr_j is linear, use sympy `solve`
                # 2. if expr_j's degree == 2, substitute to get new expr
                if degree_of_expr(expr_j) == 1:
                    results = solve((expr_i, expr_j), expr_i.free_symbols, dict=True)
                elif degree_of_expr(expr_j) == 2:
                    sym_1, sym_2 = list(expr_i.free_symbols)
                    expr_i_result = solve(expr_i, sym_1, dict=True)
                    if isinstance(expr_i_result, list):
                        expr_i_result = expr_i_result[0]
                    expr_new = expr_j.subs(expr_i_result)
                    results = solve(expr_new, sym_2, dict=True)
                else:
                    continue
                
                if len(results) == 0:
                    continue
                if isinstance(results, list):
                    results = results[0]
                    
                # add conditions
                for k, v in results.items():
                    expr_new = k - v
                    # check if already in equations
                    if simplify(Eq(expr_new, 0)) in eqs:
                        continue
                    self.problem.add('Equation', expr_new, premise, ('solve_eq', None, None))
                    self.problem.condition.simplified_equation[expr_new] = [len(self.problem.condition.items) - 1]
                    
            
            # add condition
            if len(self.problem.condition.items) - last_step > 0:
                update = True
        if update:
            self.problem.step(('solve_eq', None, None), 0)
        return 
    
    
if __name__ == '__main__':
    from utils.tools import read_json
    predicate_GDL = read_json('json/predicate_GDL_for_search.json')
    theorem_GDL = read_json('json/theorem_GDL_for_search.json')
    t_info = read_json('json/t_info_new.json')
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    solver = FormalGeoSolver(
        predicate_GDL, theorem_GDL, 
        strategy='bs', max_depth=4, beam_size=12, 
        t_info=t_info, t_freq_info=t_freq_info, 
        p_pos=None, debug=False
    )
    
    problem_data = read_json('geo_synth_2/geosynth_ENT_1_REL_1/annotations/test_1.json')
    problem_CDL = {
        "problem_id": 0,
        "construction_cdl": problem_data['construction_cdl'],
        "text_cdl": problem_data['image_cdl'],
        "image_cdl": problem_data['text_cdl'],
        "goal_cdl": problem_data['goal_cdl'],
        "problem_answer": problem_data['llm_info']['problem_answer'],
        "point_positions": problem_data['positions']
    }
    
    solver.init_search(problem_CDL)
    solver.search()
    print(solver.problem.condition.items)