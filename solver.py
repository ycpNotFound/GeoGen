import random
import time
from collections import defaultdict
from copy import copy, deepcopy
from functools import wraps

import sympy as sp
from func_timeout import FunctionTimedOut, func_timeout
from sympy import Eq, cos, rad, simplify, sin, solve, sqrt, tan
from tqdm import tqdm

from formalgeo_v2.core import EquationKillerV2 as EqKiller
from formalgeo_v2.core import GeometryPredicateLogicExecutorV2 as GPLExecutor
from formalgeo_v2.parse import (parse_predicate_gdl, parse_problem_cdl,
                             parse_theorem_gdl)
from formalgeo_v2.problem import Problem
from formalgeo_v2.tools import debug_print, get_used_pid_and_theorem
from utils.symbolic import degree_of_expr, parse_clause


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
                print(f"Function [{func.__name__}] took {elapsed_time:.3f} seconds to execute.")
            
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
        if t_name == 'cosine_theorem': 
            a = 1
        if t_info[t_name][1] == 0 or t_info[t_name][0] == 3:  # skip no used and diff t
        # if t_info[t_name][1] == 0:
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
    def __init__(self, predicate_GDL, theorem_GDL, strategy='auto', max_depth=3, beam_size=6, t_info=None, t_freq_info=None, p_pos=None, debug=False):
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

        self.leveled_condition = {}
        self.algegra_para_fail_count = {}
        self.max_algebra_iter_times = 0
        
    @make_timing_decorator('debug')
    def init_search(self, problem_CDL):
        """Initial problem by problem_CDL and build root Node."""
        EqKiller.use_cache = True  # use cache to speed up solving
        EqKiller.empty_cache()

        timing = time.time()  # timing

        self.problem = Problem(p_pos=self.p_pos)  # init problem
        self.problem.load_problem_by_fl(
            self.parsed_predicate_GDL, self.parsed_theorem_GDL, parse_problem_cdl(problem_CDL))
        # EqKiller.solve_equations(self.problem)
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
        selections = self.get_theorem_selection(
            start_step=0,
            end_step=len(self.problem.condition.items),
            sample_num_per_th=3
        )
        self.add_selections([], selections)
        self.leveled_condition[0] = {
            idx: condition for idx, condition in enumerate(
                self.problem.condition.items
            )
        }
        debug_print(self.debug, "(Depth=0)(timing={:.3f}s) Expand {} child node.".
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
                    self.problem.step(('solve_eq', None, None), 0)
        return
        

    def search(self):
        """
        Search problem and return search result.
        :return solved: <bool>, indicate whether problem solved or not.
        :return seqs: <list> of <str>, solved theorem sequences.
        self.problem.condition.items
        """

        start_step = 0
        end_step = len(self.problem.condition.items)
        for depth_i in range(1, self.max_depth+1):
            beam_count = len(self.stack)
            if self.strategy == 'auto':
                if depth_i == 0 or depth_i == 1:
                    beam_count = 300
                elif depth_i == 2:
                    beam_count = 200
                elif depth_i == 3:
                    beam_count = 100
                elif depth_i == 4:
                    beam_count = 50
                else:
                    beam_count = self.beam_size
            else:
                beam_count = self.beam_size
                    
            if len(self.stack) > beam_count:  
                stack = []
                stack = sorted(
                    self.stack, 
                    key=lambda item: self.t_freq_info[item[1][0][0]],
                    reverse=True
                )[:beam_count]
                for i in random.sample(range(len(self.stack)), beam_count):
                    stack.append(self.stack[i])
                self.stack = stack
            else:
                beam_count = len(self.stack)

            # visit all nodes in depth i
            for _ in range(beam_count):
                depth, selection = self.stack.pop(0)
                self.step_size += 1
                timing = time.time()
                self.apply_and_check_goal(selection, depth)

            # solve equations or special angles
            self.solve_special_angles()
            if depth_i % 2 == 0:
                EqKiller.solve_equations(self.problem)

            # expand new nodes in the next depth
            start_step, end_step = end_step, len(self.problem.condition.items)
            timing = time.time()
            selections = self.get_theorem_selection(
                start_step, end_step,
                sample_num_per_th=3
            )
            self.add_selections(depth_i, selections)
            debug_print(self.debug, "(Depth={})(timing={:.3f}s) Expand {} child node.".
                        format(depth_i, time.time() - timing, len(selections)))
            
            if start_step == end_step:
                # there's no more conclusions solved
                break
            if len(selections) + len(self.problem.condition.items) > 1000:
                # there's too much conclusions, stop early
                break
            else:
                # save in `leveled_condition`
                self.leveled_condition[depth_i] = {
                    idx+start_step : condition for idx, condition in enumerate(
                        self.problem.condition.items[start_step:end_step]
                    )
                }
        
        EqKiller.solve_equations(self.problem)
        return False, None
    
    def get_premise_conditions(self, condition):
        premise_ids = condition[2]
        premise_conditions = deepcopy(
            [self.problem.condition.items[i] for i in premise_ids]
        )
        return premise_conditions
    
    def search_ori(self):
        """
        Search problem and return search result.
        :return solved: <bool>, indicate whether problem solved or not.
        :return seqs: <list> of <str>, solved theorem sequences.
        self.problem.condition.items
        """
        self.leveled_condition = {}
        cur_depth = 0
        last_step = 0
        cur_step = len(self.problem.condition.items)
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
                stack = sorted(
                    self.stack, 
                    key=lambda item: self.t_freq_info[item[1][0][0]],
                    reverse=True
                )[:beam_count]
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
                    sample_num_per_th=3
                )
                self.add_selections(pos, selections)
                debug_print(self.debug, "(timing={:.4f}s) Expand {} child node.".
                            format(time.time() - timing, len(selections)))
            
            
            self.problem.check_goal()
            solved = self.problem.goal.solved
            if solved:  # solved, return result
                _, seqs = get_used_pid_and_theorem(self.problem)
                return True, seqs
            
            EqKiller.solve_equations(self.problem)
            self.solve_special_angles()
            # self.solve_equations()
            # self.solve_equation_groups()
            # self.solve_equivalence_relation()
            selections = self.get_theorem_selection()
            self.add_selections(pos, selections)
            cur_depth += 1
        
        EqKiller.solve_equations(self.problem)
        return False, None
    
    def get_theorem_selection(self, start_step, end_step, sample_num_per_th=None):
        """
        Return theorem selections according to <self.last_step>.
        :return selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        selections = []

        timing = time.time()
        related_pres = []  # new added predicates
        related_syms = []  # new added/updated equations
        # for step in range(start_step, end_step):  # get related conditions
        # for step in range(0, self.problem.condition.step_count):
        for _id in range(start_step, end_step):
            # for _id in self.problem.condition.ids_of_step[step]:
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


        logic_selections = self.try_theorem_logic(
            related_pres, 
            sample_num_per_th
        )
        algebra_selections = self.try_theorem_algebra(
            related_syms, 
            sample_num_per_th
        )
        # algebra_selections = []
        # debug_print(self.debug, "(timing={:.4f}s) Get {} algebra-related selections: {}.".format(time.time() - timing, len(algebra_selections), algebra_selections))
        # debug_print(self.debug, f'logic: {len(logic_selections)}')
        # debug_print(self.debug, f'algeb: {len(algebra_selections)}')
        
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

        # if has too much selections, do not solve for premeter / area
        pop_perimeter_area = len(selections) > 200
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
                    if pop_perimeter_area:
                        if t_para not in self.problem_a_paras:
                            selections.pop(i)

            elif "perimeter" in t_name:
                if "ratio" in t_name:
                    para1 = t_para[0:int(len(t_para) / 2)]
                    para2 = t_para[int(len(t_para) / 2):]
                    if not (para1 in self.problem_p_paras and para2 in self.problem_p_paras):
                        selections.pop(i)
                else:
                    if pop_perimeter_area:
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
                f1 = len(self.problem.condition.items_group['Square']) > 0
                f2 = len(self.problem.condition.items_group['Rhombus']) > 0
                if f1 or f2:
                    continue
                pre_selections += random.sample(v, min(sample_num_per_th, 2))
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
            'midsegment_of_triangle_property_length'
        ]
        def process_related_pres(t_name, t_branch, t_letters):
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

        if len(related_pres) > 1000 and self.debug:
            if len(self.problem.condition.items) > 300:
                related_pres = random.sample(related_pres, 300)
            with tqdm(total=len(related_pres)) as pbar:
                for t_name, t_branch, t_letters in related_pres:
                    if 'similar_arc' in t_name:
                        continue
                    process_related_pres(t_name, t_branch, t_letters)
                    pbar.update(1)
        else:
            # if len(self.problem.condition.items) > 300 and len(related_pres) > 500:
            #     related_pres = random.sample(related_pres, 500)
            for t_name, t_branch, t_letters in related_pres:
                if 'similar_arc' in t_name:
                    continue
                process_related_pres(t_name, t_branch, t_letters)

        return selections

    @make_timing_decorator('debug')
    def try_theorem_algebra(self, related_syms, sample_num_per_th=None):
        """
        Try a theorem and return can-added conclusions.
        :param related_syms: <list>, related syms.
        :return selections: <list> of ((t_name, t_branch, t_para, t_timing), ((predicate, item, premise))).
        """
        if self.max_algebra_iter_times > 500:
            sample_num_per_th = 1
        elif sample_num_per_th is None:
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

        # reduce search time
        if 'MeasureOfAngle' in paras_of_attrs:
            paras_of_attrs['MeasureOfAngle'] = random.sample(
                paras_of_attrs['MeasureOfAngle'], 
                min(len(paras_of_attrs['MeasureOfAngle']), 10)
            )
        if 'LengthOfLine' in paras_of_attrs:
            paras_of_attrs['LengthOfLine'] = random.sample(
                paras_of_attrs['LengthOfLine'], 
                min(len(paras_of_attrs['LengthOfLine']), 10)
            )

        selections = []
        iter_times = 0
        
        debug_theorems = [
            'midsegment_of_triangle_judgment_midpoint'
        ]
        for related_attr in paras_of_attrs:
            related_paras = deepcopy(set(paras_of_attrs[related_attr]))
            # down sampling
            if self.max_algebra_iter_times > 500:
                related_paras = set(random.sample(related_paras, len(related_paras)//2))

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
                    iter_times += 1
                    for letters, premise, conclusion in results:
                        theorem_para = tuple([letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]])
                        premise = tuple(premise)
                        conclusions = []
                        if t_name in debug_theorems:
                            a = 1
                        for predicate, item in conclusion:  # add conclusion
                            if self.problem.check(predicate, item, premise, t_name):
                                if predicate != "Equation":
                                    item = tuple(item)
                                conclusions.append((predicate, item, premise))

                        if len(conclusions) > 0:
                            selections.append(((t_name, t_branch, theorem_para), tuple(conclusions)))
                            # if len(selections) > self.beam_size:
                            #     return selections
        # debug_print(self.debug, f"Function [try_theorem_algebra] iter {iter_times} times.")
        if iter_times > self.max_algebra_iter_times == 0:
            self.max_algebra_iter_times = iter_times
        return selections

    def apply_and_check_goal(self, selection, depth):
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
        
        debug_theorems = [
            'midsegment_of_triangle_property_length'
        ] 
        if t_msg[0] in debug_theorems:
            a = 1
        for predicate, item, premise in conclusions:
            update = self.problem.add(predicate, item, premise, t_msg, skip_check=True) or update

        if not update:  # close current branch if applied theorem no new condition
            return None
        
        # EqKiller.solve_equations(self.problem)  # solve eq & check_goal
        
        self.problem.check_goal()
        self.problem.step(t_msg, 0)

        return self.problem.goal.solved

    def add_selections_ori(self, father_pos, selections):
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

    def add_selections(self, depth, selections):
        """
        Add selections to self.stack.
        :param father_pos: position of father branch.
        :param selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """

        for selection in selections:
            self.stack.append((depth, selection))



class Interactor:

    def __init__(self, predicate_GDL, theorem_GDL, p_pos=None, t_info=None, debug=False):
        """
        Initialize Interactor.
        :param predicate_GDL: predicate GDL.
        :param theorem_GDL: theorem GDL.
        """
        self.parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, self.parsed_predicate_GDL)
        self.p_pos = p_pos
        self.problem = None
        
        self.debug = debug
        self.last_step = 0
        self.p2t_map = get_p2t_map_fw(t_info, self.parsed_theorem_GDL)
        
        self.problem_p_paras = set()  # Perimeter
        self.problem_a_paras = set()  # Area
        self.ignore_ops = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle', 'Arc']

    def load_problem(self, problem_CDL, solve_eq=True):
        """Load problem through problem_CDL."""
        start_time = time.time()
        self.problem = Problem(p_pos=self.p_pos)
        self.problem.load_problem_by_fl(self.parsed_predicate_GDL,
                                        self.parsed_theorem_GDL,
                                        parse_problem_cdl(problem_CDL))  # load problem
        if solve_eq:
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
        # self.solve_equations()
        # self.solve_equation_groups()
        # self.solve_equivalence_relation()
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