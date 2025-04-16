import itertools
import json
import random
import re
import string
import time
from collections import defaultdict, deque
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
from sympy import (Eq, Float, Integer, Rational, Symbol, simplify, solve, sqrt,
                   total_degree)

from allocator import Allocator
from formalgeo_v2.core import EquationKillerV2 as EqKiller
from formalgeo_v2.data import DatasetLoader
from formalgeo_v2.parse import (inverse_parse_one, inverse_parse_one_theorem,
                             parse_theorem_seqs)
from formalgeo_v2.problem import Problem
from formalgeo_v2.problem.condition import Goal
from formalgeo_v2.solver import BackwardSearcher, ForwardSearcher, Interactor
from generator import ClauseGenerator
from graph import (ConditionGraph, ConditionNode, draw_graph, topological_sort,
                   topological_sort_bfs)
from plotter import Plotter
from solver import FormalGeoSolver
from utils.formulate import clause_to_nature_language, formulate_eqs_simple, sympy_to_latex
from utils.preset import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,
                          SYMBOL_MAPPING_2)
from utils.symbolic import (get_angle_measure, parse_clause,
                            replace_for_clause, subs_without_simplification)
from utils.tools import setup_seed, distance


class TargetFinder(): 
    def __init__(self, 
                 predicate_GDL,
                 theorem_GDL,
                 t_info,
                 t_freq_info,
                 allocater_states, 
                 text_cdls,
                 constr_cdls,
                 image_cdls,
                 problem_id=0,
                 replace_characters=False,
                 solver_type='formalgeo',
                 predicate_num=2,
                 debug=False):
        self.p_pos = allocater_states['p_pos']
        self.lines = allocater_states['lines']
        self.circles = allocater_states['circles']
        # self.clauses = allocater_states['clauses_base'] + allocater_states['clauses']
        # self.clauses = allocater_states['clauses']
        self.text_cdls = text_cdls
        self.constr_cdls = constr_cdls
        self.image_cdls = image_cdls
        self.predicate_GDL = predicate_GDL
        self.theorem_GDL = theorem_GDL
        self.problem_id = problem_id
        self.t_info = t_info
        self.debug = debug
        
        if replace_characters:
            self.replace_characters()
        if predicate_num == 2:
            self.max_depth = 5
            self.min_depth = 1
        elif predicate_num == 3:
            self.max_depth = 7
            self.min_depth = 2
        elif predicate_num == 4:
            self.max_depth = 9
            self.min_depth = 4
            
        self.solver = FormalGeoSolver(
            predicate_GDL,
            theorem_GDL,
            strategy="auto",
            max_depth=self.max_depth, 
            beam_size=6,
            t_info=t_info,
            t_freq_info=t_freq_info,
            p_pos=self.p_pos,
            debug=debug
        )
        
        self.problem_CDL = {
            "problem_id": problem_id,
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            # "goal_cdl": f"Value(LengthOfLine({''.join(self.lines[0])}))",
            # `goal_cdl` is not used when searching
            "goal_cdl": f"Value(q)",
            "problem_answer": "45",
        }

        if self.p_pos is not None:
            self.problem_CDL["point_positions"] = self.p_pos
        if self.lines is not None:
            self.problem_CDL["line_instances"] = self.lines
        if self.circles is not None:
            self.problem_CDL["circle_instances"] = self.circles
            
        natural_template_path = "json/predicates_to_nature_language.json"
        self.natural_template = json.load(open(natural_template_path, 'r'))
        self.symbols = []
        
    @property
    def states(self):
        return {
            "p_pos": self.p_pos,
            "lines": self.lines,
            "circles": self.circles,
            # "clauses": self.clauses
        }
        
    def add_new_symbol(self):
        for c in ['x', 'y', 'z', 'a', 'b', 'c']:
            if c not in self.symbols:
                return Symbol(c)
        
    def replace_characters(self):
        if self.debug:
            start_char = 'A'
        else:
            start_char = random.choice(['A', 'E', 'I', 'M', 'R'])
        start_idx = string.ascii_uppercase.index(start_char)
        chars = string.ascii_uppercase[start_idx: start_idx + len(self.p_pos)]
        mapping = {}
        for point_i, char in zip(self.p_pos, chars):
            mapping[point_i] = char
            
        # points
        self.p_pos = {mapping[k]: self.p_pos[k] for k in self.p_pos}
        # lines / circles
        self.lines = [tuple([mapping[p] for p in l]) for l in self.lines]
        self.circles = [mapping[p] for p in self.circles]
        # self.points_on_circle = {mapping[k]: [mapping[x] for x in v] 
        #                          for k, v in self.points_on_circle.items()}
        # clauses
        # self.clauses = [replace_for_clause(c, mapping) for c in self.clauses]
        self.text_cdls = [replace_for_clause(c, mapping) for c in self.text_cdls]
        self.constr_cdls = [replace_for_clause(c, mapping) for c in self.constr_cdls]
        self.image_cdls = [replace_for_clause(c, mapping)for c in self.image_cdls]
        return
    
    @staticmethod
    def targets_filter_1(conditions_to_sample, value_of_sym, strict=False):
        # for potential calculation target: 
        # 1. only has <= 2 vars (<=3 for lines) | strict: only has 1 var
        # 2. only has linear term, degree <= 2
        # 3. if has 2 vars, can not be both solved value
        # 4. only has symbols begin with 'll_' or 'ma_', or only has one symbol
        # 5. remove angle / arc measure that >= 180
        # 6. do not have symbols like 'rst', 'rmt', 'rsq', 'rmq', 'rsa', 'ht', 'hq'
        new_targets = []
        for condition in conditions_to_sample:
            if condition[3][0] == 'prerequisite':
                continue
            if condition[-1] == 0: # 0 step
                continue
            if condition[0] == 'Equation':
                syms = [str(x) for x in list(condition[1].free_symbols)]
                f1 = len(syms) <= 2 or \
                    (len(syms) <= 3 and \
                    all(['ll_' in sym for sym in syms]))
                if strict:
                    f1 = len(syms) == 1
                try:
                    if strict:
                        f2 = condition[1].as_poly(*list(condition[1].free_symbols)).total_degree() < 2
                    else:
                        f2 = condition[1].as_poly(*list(condition[1].free_symbols)).total_degree() <= 2
                except:
                    f2 = False
                f3 = True
                if len(syms) == 2:
                    if all([
                        value_of_sym[sym] is not None 
                        for sym in list(condition[1].free_symbols)
                    ]):
                        f3 = False
                        
                f4 = all(['ll_' in sym or 'ma_' in sym for sym in syms]) or len(syms) == 1

                f5 = True
                if len(syms) == 1:
                    if 'ma_' in syms[0] or 'mar_' in syms[0]:
                        f5 = abs(condition[1].as_coefficients_dict().get(1, 0)) < 180

                f6 = True
                if len(condition[1].free_symbols) == 1:
                    sym = condition[1].free_symbols.pop()
                    f6 = str(sym).split('_')[0] not in ['rst', 'rmt', 'rsq', 'rmq', 'rsa', 'ht', 'hq']
  
                if all([f1, f2, f3, f4, f5, f6]):
                    new_targets.append(condition)
                    
            elif condition[0] in PREDICATES_REL + PREDICATES_REL_2 +  PREDICATES_ENT + ['Collinear', 'Cocircular']:
                new_targets.append(condition)

        return new_targets
    
    @staticmethod
    def targets_filter_2(new_targets, strict=False):
        # if has too many 'solve_eq' targets like 'a+b-90=0', randon sample to 5
        # if strict: remove all targets like 'a+b-90=0'
        solve_eq_targets = [
            t for t in new_targets if t[0] == 'Equation'
            and len(t[1].free_symbols) >= 2
            and t[1].as_coefficients_dict().get(1, 0) != 0
        ]
        if strict:
            for t in solve_eq_targets:
                new_targets.remove(t)
            solve_eq_targets_2 = [
                t for t in new_targets if t[0] == 'Equation'
                and len(t[1].free_symbols) >= 3
            ]
            for t in solve_eq_targets_2:
                new_targets.remove(t)
        
        elif len(solve_eq_targets) > 5:
            targets_to_delete = random.sample(solve_eq_targets, len(solve_eq_targets) - 5)
            for t in targets_to_delete:
                new_targets.remove(t)
        

        return new_targets


    @staticmethod
    def targets_filter_3(new_targets, theorems_for_targets, max_depth, strict=False):
        # 1. sort by len of theorems (more but can not include too many 'solve_eq')
        # 2. sort by num of unsolved symbols (less)
        # 3. sort by token diversity of theorems (more)
        def filter_idx(target, theorem_list):
            score_1 = len(theorem_list) 
            solve_eq_cnt = sum([1 if item == 'solve_eq' else 0 for item in theorem_list])
            geo_theorem_cnt = len(theorem_list) - solve_eq_cnt
            if solve_eq_cnt >= 4 or len(theorem_list) >= max_depth + 2 or solve_eq_cnt >= geo_theorem_cnt + 2:
                score_1 = 0
                
            score_2 = -1
            if target[0] == 'Equation':
                score_2 = - len(target[1].free_symbols)
                # treat constant term as symbols
                if target[1].as_coefficients_dict().get(1, 0) != 0:
                    score_2 -= 1
                
            token_set = set()
            for item in theorem_list:
                tokens = item.split('(')[0].split('_')
                token_set.update(tokens)
            score_3 = len(token_set)
            return (score_1, score_2, score_3)
        
        new_targets = sorted(
            theorems_for_targets, 
            key=lambda k: filter_idx(
                k, 
                theorems_for_targets[k],
            ), 
            reverse=True
        )
        idx_for_targets = [filter_idx(k, theorems_for_targets[k]) for k in new_targets]
        return new_targets, idx_for_targets
    
    @staticmethod
    def targets_into_groups(new_targets, strict=False):
        # the third filter: 
        # classified into groups according to target type
        # choose only 2 targets for each 'prove' predicate name
        targets_angle = [t for t in new_targets 
                         if 'Equation'== t[0] and 'ma_' in str(t[1])]
        targets_line = [t for t in new_targets 
                        if 'Equation'== t[0] and 'll_' in str(t[1])]
        targets_value = [t for t in new_targets
                         if 'Equation'== t[0] and len(t[1].free_symbols) == 1 and 
                         'll_' not in str(t[1].free_symbols.pop()) and 
                         'ma_' not in str(t[1].free_symbols.pop())]
        targets_prove = [t for t in new_targets
                         if 'Equation' != t[0]]
        prove_cnt_dict = defaultdict(int)
        _targets_prove = []
        # filter prove target
        for t in targets_prove:
            prove_cnt_dict[t[0]] += 1
            if strict:
                if prove_cnt_dict[t[0]] <= 1:
                    _targets_prove.append(t)
            else:
                if prove_cnt_dict[t[0]] <= 2:
                    _targets_prove.append(t)
        targets_prove = _targets_prove

        # filter value target:
        # may has exprs correspond to same simplified expr: x-2 <=> (x-2)^2
        # select into groups and choose targets with mini str len
        sim_expr_2_targets = {}
        for t in targets_value:
            sym = t[1].free_symbols.pop()
            
            eq = Eq(t[1], 0)
            res = solve(eq, dict=True)
            if len(res) != 0:
                new_expr = sym - res[0][sym]
                if str(new_expr) not in sim_expr_2_targets:
                    sim_expr_2_targets[str(new_expr)] = []
                sim_expr_2_targets[str(new_expr)].append(t)
        _targets_value = []
        for expr_str, t_list in sim_expr_2_targets.items():
            chosen_t = min(t_list, key=lambda x: len(str(x[1])))
            _targets_value.append(chosen_t)
        targets_value = _targets_value

        # return grouped targets
        targets_dict = {
            "line": targets_line,
            "angle": targets_angle,
            "value": targets_value,
            "prove": targets_prove
        }
        return targets_dict
    
    def get_conditions_to_sample(self):
        # determine the min / max depth of conditions to sample
        max_depth = max([k for k, v in self.solver.leveled_condition.items() if len(v) != 0])
        max_depth = min([max_depth, self.max_depth])
        min_depth = max(self.min_depth, max_depth - 2)
        start_idx = len(self.solver.problem.condition.items)
        end_idx = len(self.solver.problem.condition.items)

        # determine the range of conditions to sample
        for k, v in self.solver.leveled_condition.items():
            if k == min_depth:
                idx_list = list(v.keys())
                start_idx = min(min(idx_list), start_idx)
        
        # if too few conditions, decrease min_depth
        if end_idx - start_idx < 10 and max_depth > 3:
            min_depth = max(self.min_depth, max_depth - 1) 
            for k, v in self.solver.leveled_condition.items():
                if k == min_depth:
                    idx_list = list(v.keys())
                    start_idx = min(min(idx_list), start_idx)
                    
        conditions_to_sample = self.solver.problem.condition.items[start_idx:end_idx]

        # add solved values
        for sym, value in self.solver.problem.condition.value_of_sym.items():
            if value is not None:
                condition_idx = self.solver.problem.condition.get_id_by_predicate_and_item('Equation', sym - value)
                condition_item = self.solver.problem.condition.items[condition_idx]
                if condition_item not in conditions_to_sample:
                    conditions_to_sample.append(condition_item)
        return conditions_to_sample
    
    def filter_conditions(self, condition_graph: ConditionGraph, strict=False):
        conditions_to_sample = self.get_conditions_to_sample()
        # filter 1
        new_targets = self.targets_filter_1(
            conditions_to_sample,
            self.solver.problem.condition.value_of_sym,
            strict=strict
        )
        # filter 2, if has too many targets, filter by more strict rules
        new_targets = self.targets_filter_2(
            new_targets,
            strict=strict
        )
            
        # find solution / theorems for each target
        theorems_for_targets = {}
        solution_for_targets = {}
        solution_dict_for_targets = {}
        # draw_graph(condition_graph, 'test', new_targets[11], img_dir='imgs_test')
        for target in new_targets:
            (
                solution_str,
                solution_formal_dict,
                theorems, 
                sub_nodes,
                too_complex_flag
            ) = self.find_solution_for_target(
                self.solver.problem,
                condition_graph, 
                target, 
                self.natural_template,
                self.solver.parsed_theorem_GDL
            )
            if too_complex_flag:
                continue

            theorems_for_targets[target] = [item['theorem'] for item in solution_formal_dict.values() if item['theorem'] is not None]
            solution_for_targets[target] = solution_str
            solution_dict_for_targets[target] = solution_formal_dict

            # level = sum([1 if item['theorem'] is not None else 0 for item in solution_formal_dict.values()])

        if self.debug:
            _solution_for_targets = {
                str(k): v.split('\n') for k, v in
                solution_for_targets.items()
            }
            with open('json/solution_test.json', 'w', encoding='utf-8') as f:
                json.dump(_solution_for_targets, f, indent=4, ensure_ascii=False)
        
        # filter 3, sort by rules
        new_targets, idx_for_targets = self.targets_filter_3(
            new_targets, theorems_for_targets, self.max_depth, strict=strict
        )
        # formulate into groups by target type
        targets_dict = self.targets_into_groups(new_targets)
        return targets_dict, theorems_for_targets, solution_for_targets, solution_dict_for_targets
    
    def find_target_and_solution(self, condition_graph: ConditionGraph, strict: bool):
        (
            targets_dict,
            theorems_for_targets,
            solution_for_targets,
            solution_dict_for_targets
        ) = self.filter_conditions(condition_graph, strict=strict)
        types_to_choose = [k for k in targets_dict if len(targets_dict[k]) != 0]
        if len(types_to_choose) == 0:
            return None, None, None, None, None, None, None
        
        # random choose target type
        weight_on_types = [np.log(2*len(targets_dict[t])) for t in types_to_choose]
        prob_on_types = [w / sum(weight_on_types) for w in weight_on_types]
        target_type = random.choices(types_to_choose, weights=prob_on_types, k=1)[0]
        chosen_targets = targets_dict[target_type]
        
        # random choose target (from top-8)
        chosen_target = random.choice(chosen_targets[:8])
        chosen_thoerems = theorems_for_targets[chosen_target]
        chosen_solution = solution_for_targets[chosen_target]
        chosen_solution_dict = solution_dict_for_targets[chosen_target]
        # problem_level = depth_for_targets[chosen_target]
        problem_level = len(chosen_thoerems)
        _ = self.find_solution_for_target(
            self.solver.problem,
            condition_graph,
            chosen_target,
            self.natural_template,
            self.solver.parsed_theorem_GDL
        )
        available_targets = []
        for k, v in targets_dict.items():
            for t in v:
                available_targets.append(t)
        available_num = len(available_targets)
        
        return target_type, chosen_target, problem_level, chosen_solution, chosen_solution_dict, chosen_thoerems, available_num

    @staticmethod
    def find_solution_for_target(
            problem: Problem,
            condition_graph: ConditionGraph, 
            target_condition: Tuple,
            natural_template: Dict,
            parsed_theorem_GDL: Dict,
            expand_flag: bool = False
        ):
        too_complex = False
        sub_nodes, sub_nodes_adj_table = condition_graph.backward_construct_sub_graph([target_condition])
    
        # sort by index of applying theorem 
        sub_nodes = topological_sort(sub_nodes, sub_nodes_adj_table)
        # sub_nodes = topological_sort_bfs(sub_nodes, sub_nodes_adj_table)
        
        sub_nodes_idx = [n.idx for n in sub_nodes]
        sub_nodes_clauses = [
            inverse_parse_one(n.value[0], n.value[1], problem)
            for n in sub_nodes
        ]
        # inverse_parse_one('Equation', Symbol('rst_bcaabd')-sqrt(2), problem)
        sub_nodes_statements = clause_to_nature_language(
            sub_nodes_clauses, 
            natural_template,
            upper=False,
            replace_sym=True,
            replace_sym_mode='math'
        )
        pred_ignore = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle', 'Arc', 'Circle', 'Free']
        extend_nodes = [n for n in sub_nodes if n.value[3][0] == 'extended' and n.value[0] not in pred_ignore]

        theorems_formal = []
        solution_str = "Solution: "
        solution_formal_dict = {}
        step_count = 0
        sub_nodes_by_step = {}
        last_theorem = None
        last_premise_ids = None
        # special token: 
        # <by> - theorem
        # <because> - premise condition
        # <therefore> - extend condition
        
        for i, node in enumerate(sub_nodes):
            theorem = node.value[3][0]
            predicate = node.value[0]
            statement = sub_nodes_statements[i]
            clause_i = sub_nodes_clauses[i]
            
            if theorem not in ['prerequisite', 'extended', 'solve_eq']:
                theorems_formal.append(
                    inverse_parse_one_theorem(node.value[3], parsed_theorem_GDL)
                )
                
            if theorem == 'prerequisite': 
                # add all extended condition
                extend_conditions = []
                # add all nodes in current step
                extend_nodes_i = []
                queue = deque([n for n in extend_nodes if node.idx in n.value[2]])
                while queue:
                    extend_node = queue.popleft()
                    extend_idx = sub_nodes.index(extend_node)
                    extend_statement = sub_nodes_statements[extend_idx]
                    extend_predicate = extend_node.value[0]
                    if predicate == extend_predicate and predicate in ['Collinear', 'Cocircular']:
                        continue
                    extend_conditions.append(extend_statement)
                    extend_nodes_i.append(extend_node)
                    for n in extend_nodes:
                        if extend_node.idx in n.value[2]:
                            queue.append(n)
                if len(extend_conditions) != 0:
                    step_count += 1
                    sub_nodes_by_step[step_count] = extend_nodes_i
                    solution_str += f"\n{step_count}. <because> {statement}, <therefore>"
                    solution_formal_dict[step_count] = {
                        "theorem": None,
                        "condition": [clause_i],
                        "conclusion": [sub_nodes_clauses[sub_nodes_statements.index(c)] for c in extend_conditions]
                    }
                    
                    f"{step_count} | None | {clause_i} |"
                    for c in extend_conditions:
                        solution_str += f'\n- {c}.'
                    
            elif theorem == 'extended': 
                # add extended condition in 'prerequisite' or other theorems
                pass
                
            else: # using theorem 
                # find all premise statements grouped by step
                premise_statements = {}
                premise_clauses = {}
                for premise_idx in node.value[2]:
                    if premise_idx not in sub_nodes_idx:
                        continue
                    # ignore condition like Line(AB)
                    premise_node = sub_nodes[sub_nodes_idx.index(premise_idx)]
                    if premise_node.value[0] in pred_ignore:
                        continue
                    premise_step = next((k for k, v in sub_nodes_by_step.items() if premise_node in v), None)
                    premise_statement = sub_nodes_statements[sub_nodes_idx.index(premise_idx)]
                    premise_clause = sub_nodes_clauses[sub_nodes_idx.index(premise_idx)]
                    if premise_step is None:
                        if 'given condition' not in premise_statements:    
                            premise_statements['given condition'] = [premise_statement]
                            premise_clauses['given condition'] = [premise_clause]
                        else:
                            premise_statements['given condition'].append(premise_statement)
                            premise_clauses['given condition'].append(premise_clause)
                    else:
                        if f"step {premise_step}" not in premise_statements:
                            premise_statements[f"step {premise_step}"] = [premise_statement]
                            premise_clauses[f"step {premise_step}"] = [premise_clause]
                        else:
                            premise_statements[f"step {premise_step}"].append(premise_statement)
                            premise_clauses[f"step {premise_step}"].append(premise_clause)

                # add theorem first
                # if use the same theorem and same premise condition
                # append conclusion in the previous step
                f1 = theorem != last_theorem
                f2 = node.value[2] != last_premise_ids
                f3 = len(premise_statements) != 0
                f4 = theorem == 'solve_eq'
                if f4:
                    step_count += 1
                    solution_str += f'\n{step_count}. Solve equations:\n'
                    eq_solution = formulate_eqs_simple(premise_statements)
                    
                    # check if solve too many eqs in one step
                    if eq_solution is None: 
                        too_complex = True
                        break
                    solution_str += eq_solution
                    solution_formal_dict[step_count] = {
                        "theorem": "solve_eq",
                        "condition": [item for sublist in premise_clauses.values() for item in sublist]
                    }
                        
                elif f1 or (f2 and f3):
                    step_count += 1
                    solution_str += f'\n{step_count}. <by> {theorem}, '
                    solution_formal_dict[step_count] = {
                        "theorem": theorem,
                        "condition": []
                    }

                    # add all premise statements
                    if len(premise_statements) != 0:
                        solution_str += '<because> '
                        if len(premise_statements) == 1:
                            k = list(premise_statements.keys())[0]
                            v = premise_statements[k]
                            solution_str += f"{', '.join(v)} from {k}, "
                            premise_clauses = [sub_nodes_clauses[sub_nodes_statements.index(s)] for s in v]
                            solution_formal_dict[step_count]['condition'] += premise_clauses
                            # solution_formal_str += f"{' '.join(premise_clauses)}"
                        else:
                            for k, v in premise_statements.items():
                                solution_str += f"{', '.join(v)} from {k}, "
                                premise_clauses = [sub_nodes_clauses[sub_nodes_statements.index(s)] for s in v]
                                solution_formal_dict[step_count]['condition'] += premise_clauses
                                # solution_formal_str += f"{' '.join(premise_clauses)}"

                        
                extend_conditions = [statement]
                extend_nodes_i = [node]
                extend_clause = [clause_i]
                # add all extended conditions
                queue = deque([n for n in extend_nodes if node.idx in n.value[2]])
                while queue:
                    extend_node = queue.popleft()
                    extend_idx = sub_nodes.index(extend_node)
                    extend_nodes_i.append(extend_node)
                    extend_conditions.append(sub_nodes_statements[extend_idx])
                    extend_clause.append(sub_nodes_clauses[extend_idx])
                    for n in extend_nodes:
                        if extend_node.idx in n.value[2]:
                            queue.append(n)
                
                extend_str= '\n- '.join(extend_conditions)
                # extend_clause_str = ' '.join(extend_clause)
                if not (f1 or (f2 and f3)) and not f4:
                    # same theorem and no premise statements
                    solution_str += f'\n- {extend_str}. '
                    sub_nodes_by_step[step_count] += extend_nodes_i
                    solution_formal_dict[step_count]['conclusion'] = extend_clause
                    # solution_formal_str += f' {extend_clause_str}'
                else:
                    solution_str += f"<therefore>\n- {extend_str}. "
                    sub_nodes_by_step[step_count] = extend_nodes_i
                    # solution_formal_str += f'| {extend_clause_str}'
                    solution_formal_dict[step_count]['conclusion'] = extend_clause
                
                last_theorem = theorem
                last_premise_ids = node.value[2]
            
        return solution_str, solution_formal_dict, theorems_formal, sub_nodes, too_complex
    
    
    def distance(self, line):
        p1, p2 = line
        x1, y1 = self.p_pos[p1]
        x2, y2 = self.p_pos[p2]
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 )
    
    def create_question_three_line_symbols(self, target):
        target_sym = random.choice(list(target[1].free_symbols))
        target_line = str(target_sym).split('_')[-1].upper()
        sym_1, sym_2 = list(target[1].free_symbols - set([target_sym]))
        line_1 = str(sym_1).split('_')[-1].upper()
        line_2 = str(sym_2).split('_')[-1].upper()

        len_1 = int(self.distance(line_1))
        len_2 = int(self.distance(line_2))
        if len_1 == len_2:
            if self.distance(line_1) < self.distance(line_2):
                len_1 -= 1
            else:
                len_2 -= 1 
        expr = target[1].subs({sym_1: len_1, sym_2: len_2})
        res = solve(Eq(expr, 0))
        if len(res) == 0:
            return None
        target_value = res[0]
        if 'sqrt' in str(target_value):
            target_value = re.sub(r'sqrt\(([^)]+)\)', r'√\1', str(target_value))
            
        target_str = f"find the length of {target_line}"
        target_cdl = f"Value(LengthOfLine({target_line}))"
        add_cdls = [
            f"Equal(LengthOfLine({line_1}),{len_1})",
            f"Equal(LengthOfLine({line_2}),{len_2})",
        ]
        add_conditions = [
            f"{line_1} = {len_1}", f"{line_2} = {len_2}"
        ]
        conclusion = f"{target_line} = {target_value}"
        res_info = {
            "conclusion": conclusion,
            "add_cdls": add_cdls,
            "add_conditions": add_conditions,
            "target_value": target_value,
            "target_str": target_str,
            "target_cdl": target_cdl
        }
        return res_info
    
    def create_question_two_line_symbols(self, target):
        target_sym = random.choice(list(target[1].free_symbols))
        other_sym = list(target[1].free_symbols - set([target_sym]))[0]
        target_line = str(target_sym).split('_')[-1].upper()
        other_line = str(other_sym).split('_')[-1].upper()
        # assign other_sym to value
        flag_1 = random.choice([True, False])
        flag_sqrt = 'sqrt' in str(target[1])
        if flag_1 or flag_sqrt and self.solver.problem.p_pos is not None:
            value = random.randint(1, 10)
            expr = target[1].subs({other_sym: value})
            res = solve(Eq(expr, 0))
            if len(res) == 0:
                return None
            target_value = res[0]
            target_str = f"find the length of {target_line}"
            target_cdl = f"Value(LengthOfLine({target_line}))"
            add_cdls = [f"Equal(LengthOfLine({other_line}),{value})"]
            add_conditions = [f"{other_line} = {value}"]
            conclusion = f"{target_line} = {target_value}"
        else:
            v1 = random.randint(1, 10)
            v2 = random.randint(1, 10)
            v3 = random.randint(1, 10)
            v4 = random.randint(1, 10)
            if v1 == v3:
                v3 = v3 + random.randint(1, 5)
            if v2 == v4:
                v4 = v4 + random.randint(1, 5)
            new_sym = self.add_new_symbol()
            expr_1 = v1*new_sym + v2
            expr_2 = v3*new_sym + v4
            expr_fixed = subs_without_simplification(
                target[1], 
                {target_sym: expr_1, other_sym: expr_2}
            )
            expr = target[1].subs({target_sym: expr_1, other_sym: expr_2})
            res = solve(Eq(expr, 0))
            if len(res) == 0:
                return None
            target_value = res[0]
            target_str = f"find the value of {str(new_sym)}"
            target_cdl = f"Value({str(new_sym)})"
            add_cdls = [
                f"Equal(LengthOfLine({target_line}),{str(expr_1)})",
                f"Equal(LengthOfLine({other_line}),{str(expr_2)})",
            ]
            add_conditions = [
                f"{target_line} = {str(expr_1)}",
                f"{other_line} = {str(expr_2)}",
            ]
            conclusion = f"\n- $ {expr_fixed} = 0 $.\n- $ {expr} = 0 $.\n- $ {new_sym} = {target_value} $"
        
        res_info = {
            "conclusion": conclusion,
            "add_cdls": add_cdls,
            "add_conditions": add_conditions,
            "target_value": target_value,
            "target_str": target_str,
            "target_cdl": target_cdl
        }
        return res_info
    
    def create_question_two_angle_symbols(self, target):
        sym_1, sym_2 = list(target[1].free_symbols)
        angle_1 = str(sym_1).split('_')[-1].upper()
        angle_2 = str(sym_2).split('_')[-1].upper()
        angle_1_val = self.solver.problem.condition.value_of_sym[sym_1]
        angle_2_val = self.solver.problem.condition.value_of_sym[sym_2]
        flag_1 = type(angle_1_val) in [Integer, Float]
        flag_2 = type(angle_2_val) in [Integer, Float]
        if flag_1 and flag_2: # angle_1 and angle_2 both solved
            return self.create_question_prove(target)
        if flag_1 and not flag_2: # angle_1 = n (solved by symbolic), solve angle_2
            other_value = angle_1_val
            expr = target[1].subs({sym_1: other_value})
            res = solve(Eq(expr, 0))
            if len(res) == 0:
                return None
            target_value = res[0]
            target_str = f"find the measure of \\angle {angle_2}"
            target_cdl = f"Value(MeasureOfAngle({angle_2}))"
            add_conditions = []
            add_cdls = []
            conclusion = f"\\angle {angle_2} = {target_value}°"
        elif flag_2 and not flag_1: # angle_2 = n (solved by symbolic), solve angle_1
            other_value = angle_2_val
            expr = target[1].subs({sym_2: other_value})
            res = solve(Eq(expr, 0))
            if len(res) == 0:
                return None
            target_value = res[0]
            target_str = f"find the measure of \\angle {angle_1}"
            target_cdl = f"Value(MeasureOfAngle({angle_1}))"
            add_conditions = []
            add_cdls = []
            conclusion = f"\\angle {angle_1} = {target_value}°"

        elif not flag_1 and not flag_2:

            if hasattr(self.solver.problem, 'p_pos') and self.solver.problem.p_pos is not None:
                # calculate value of one angle from coordinate, solve the other
                target_angle = random.choice([angle_1, angle_2])
                other_angle = list(set([angle_1, angle_2]) - set([target_angle]))[0]
                other_pos = [self.p_pos[p] for p in other_angle]
                other_value = get_angle_measure(other_pos[1], other_pos[2], other_pos[0])
                
                other_sym = sym_1 if other_angle == angle_1 else sym_2
                expr = target[1].subs({other_sym: other_value})
                res = solve(Eq(expr, 0))
                if len(res) == 0:
                    return None
                target_value = res[0]
                target_str = f"find the measure of \\angle {target_angle}"
                target_cdl = f"Value(MeasureOfAngle({target_angle}))"
                add_conditions = [f"\\angle {other_angle} = {other_value}°"]
                add_cdls = [f"Equal(MeasureOfAngle({other_angle}),{other_value})"]
                conclusion = f"\\angle {target_angle} = {target_value}°"

            else:
                # angle_1, angle_2 have no value, solve for new sym or prove
                flag_3 = random.choice([True, False]) 
                if flag_3: # angle_1 = ax+b, angle_2 = cx+d, solve x
                    v1 = random.randint(1, 10)
                    v2 = random.randint(1, 10)
                    v3 = random.randint(1, 10)
                    v4 = random.randint(1, 10)
                    if v1 == v3:
                        v3 = v3 + random.randint(1, 5)
                    new_sym = self.add_new_symbol()
                    expr_1 = v1*new_sym + v2
                    expr_2 = v3*new_sym + v4
                    expr = target[1].subs({sym_1: expr_1, sym_2: expr_2})
                    expr_fixed = subs_without_simplification(
                        target[1], 
                        {sym_1: expr_1, sym_2: expr_2}
                    )
                    res = solve(Eq(expr, 0))
                    if len(res) == 0:
                        return None
                    target_value = res[0]
                    target_str = f"find the value of {str(new_sym)}"
                    target_cdl = f"Value({str(new_sym)})"
                    add_cdls = [
                        f"Equal(MeasureOfAngle({angle_1}),{str(expr_1)})",
                        f"Equal(MeasureOfAngle({angle_2}),{str(expr_2)})",
                    ]
                    add_conditions = [
                        f"\\angle {angle_1} = {str(expr_1)}",
                        f"\\angle {angle_2} = {str(expr_2)}",
                    ]
                    conclusion = f"\n- $ {expr_fixed} = 0 $.\n- $ {expr} = 0 $.\n- $ {new_sym} = {target_value} $"
                else: 
                    return self.create_question_prove(target)

            
        res_info = {
            "conclusion": conclusion,
            "add_cdls": add_cdls,
            "add_conditions": add_conditions,
            "target_value": target_value,
            "target_str": target_str,
            "target_cdl": target_cdl
        }
        return res_info
    
    def create_question_one_symbol(self, target, angle_ids={}):
        sym = list(target[1].free_symbols)[0]
        
        res = solve(Eq(target[1], 0))
        if len(res) == 0:
            return None
        target_value = res[0]
        ps = str(sym).split('_')[-1].upper()

        if 'll_' in str(sym):
            conclusion = f"{ps} = {target_value}"
            target_str = f"find the length of {ps}"
            target_cdl = f"Value(LengthOfLine({ps}))"
        elif 'ma_' in str(sym):
            conclusion = f"\\angle {ps} = {target_value}°"
            
            angle_id = next((k for k in angle_ids if ps in angle_ids[k]), None)
            if angle_id is None:
                target_str = f"find the measure of \\angle {ps}"
            else:
                target_str = f"find the measure of \\angle {angle_id}"
            target_cdl = f"Value(MeasureOfAngle({ps}))"
        elif 'pt_' in str(sym):
            conclusion = f"perimeter of \\triangle {ps} = {target_value}"
            target_str = f"find the perimeter of \\triangle {ps}"
            target_cdl = f"Value(PerimeterOfTriangle({ps}))"
        elif 'at_' in str(sym):
            conclusion = f"area of \\triangle {ps} = {target_value}"
            target_str = f"find the area of \\triangle {ps}"
            target_cdl = f"Value(AreaOfTriangle({ps}))"
        elif 'ht_' in str(sym):
            conclusion = f"height of \\triangle {ps} = {target_value}"
            target_str = f"find the height of \\triangle {ps}"
            target_cdl = f"Value(HeightOfTriangle({ps}))"
        elif 'pq_' in str(sym):
            conclusion = f"perimeter of {ps} = {target_value}"
            target_str = f"find the perimeter of {ps}"
            target_cdl = f"Value(PerimeterOfQuadrilateral({ps}))"
        elif 'aq_' in str(sym):
            conclusion = f"area of {ps} = {target_value}"
            target_str = f"find the area of {ps}"
            target_cdl = f"Value(AreaOfQuadrilateral({ps}))"
        elif 'hq_' in str(sym):
            conclusion = f"height of {ps} = {target_value}"
            target_str = f"find the area of {ps}"
            target_cdl = f"Value(AreaOfQuadrilateral({ps}))"
        elif 'la_' in str(sym):
            conclusion = f"legnth of \\arc {ps[1:]} = {target_value}"
            target_str = f"find the legnth of {ps[1:]}"
            target_cdl = f"Value(LengthOfArc({ps}))"
        elif 'mar_' in str(sym):
            conclusion = f"measure of \\arc {ps[1:]} = {target_value}"
            target_str = f"find the measure of {ps[1:]}"
            target_cdl = f"Value(MeasureOfArc({ps}))"
        elif 'rc_' in str(sym):
            conclusion = f"radius of \\odot {ps} = {target_value}"
            target_str = f"find the radius of \\odot {ps}"
            target_cdl = f"Value(RadiusOfCircle({ps}))"
        elif 'dc_' in str(sym):
            conclusion = f"diameter of \\odot {ps} = {target_value}"
            target_str = f"find the diameter of \\odot {ps}"
            target_cdl = f"Value(DiameterOfCircle({ps}))"
        elif 'pc_' in str(sym):
            conclusion = f"perimeter of \\odot {ps} = {target_value}"
            target_str = f"find the perimeter of \\odot {ps}"
            target_cdl = f"Value(PerimeterOfCircle({ps}))"
        elif 'ac_' in str(sym):
            conclusion = f"area of \\arc {ps} = {target_value}"
            target_str = f"find the area of {ps}"
            target_cdl = f"Value(AreaOfCircle({ps}))"
        elif 'ps_' in str(sym):
            conclusion = f"perimeter of sector {ps} = {target_value}"
            target_str = f"find the perimeter of sector {ps}"
            target_cdl = f"Value(PerimeterOfSector({ps}))"
        elif 'as_' in str(sym):
            conclusion = f"area of sector {ps} = {target_value}"
            target_str = f"find the area of sector {ps}"
            target_cdl = f"Value(AreaOfSector({ps}))"

        else:
            sym = str(sym)
            conclusion = f"{sym} = {target_value}"
            target_str_list = [
                f"find the value of {sym}",
                f"find {sym}"
            ]
            target_str = random.choice(target_str_list)
            target_cdl = f"Value({sym})"
            
        res_info = {
            "conclusion": conclusion,
            "add_cdls": [],
            "add_conditions": [],
            "target_value": target_value,
            "target_str": target_str,
            "target_cdl": target_cdl
        }
        return res_info
    
    def create_question_prove(self, target):
        target_value = str(target[1])
        if target[0] in PREDICATES_ENT:
            conclusion = self.natural_template[target[0]][0].format(
                points=''.join(target[1]))
            clause = f"{target[0]}({''.join(target[1])})"
        elif target[0] in PREDICATES_REL + PREDICATES_REL_2:
            clause = self.target_tuple_to_clause(target)
            _, items = parse_clause(clause)
            conclusion = self.natural_template[target[0]][0].format(
                p1=items[0], p2=items[1])
        elif target[0] == 'Collinear':
            clause = f"Collinear({''.join(target[1])})"
            conclusion = f"{', '.join(target[1])} is collinear"
        elif target[0] == 'Cocircular':
            clause = f"Cocircular({target[1][0]},{''.join(target[1][1:])})"
            conclusion = f"{', '.join(target[1][1:])} is on circle {target[1][0]}"
        elif target[0] == 'Equation':
            clause = f"Equation({target[1]})"
            conclusion = f"{target[1]} = 0"
            conclusion = sympy_to_latex(conclusion)
        else:
            raise KeyError(target[0])
        
        target_str = f'prove that {conclusion}'
        if target[0] == 'Equation':
            target_cdl = clause
        else:
            target_cdl = f"Relation({clause})"
        
        res_info = {
            "conclusion": conclusion,
            "add_cdls": [],
            "add_conditions": [],
            "target_value": target_value,
            "target_str": target_str,
            "target_cdl": target_cdl
        }
        return res_info
    


    def create_question(self, target: Tuple, used_symbols=None, angle_ids={}):
        self.symbols = used_symbols if used_symbols is not None else []
        # create target and added conditions
        problem_text = random.choice([
            "",
            "In this figure, ",
            "As shown in the figure, ",
            "According to the diagram, ",
            "Based on the figure provided, "
        ])

        conditions = clause_to_nature_language(
            # self.problem_CDL['text_cdl'] + self.problem_CDL['image_cdl'],
            self.problem_CDL['text_cdl'] + self.problem_CDL['construction_cdl'] + self.problem_CDL['image_cdl'],
            self.natural_template,
            upper=False,
            replace_sym=True,
            replace_sym_mode='math'
        )

        # information to return
        # conclusion, add_cdls, add_conditions 
        # target_value, target_str, target_cdl
        # target[1] like: - ll_cd + ll_ed, ma_abc + ma_edf - 180
        if target[0] == 'Equation': 
            if len(target[1].free_symbols) == 3 and self.solver.problem.p_pos is not None:
                res_info = self.create_question_three_line_symbols(target)
            elif len(target[1].free_symbols) == 2:
                sym_str = str(list(target[1].free_symbols)[0])
                if 'll' in sym_str:
                    res_info = self.create_question_two_line_symbols(target)
                    
                elif 'ma' in sym_str:
                    res_info = self.create_question_two_angle_symbols(target)
                else:
                    res_info = self.create_question_prove(target)
                    
            elif len(target[1].free_symbols) == 1:
                res_info = self.create_question_one_symbol(target, angle_ids=angle_ids)
            else:
                res_info = self.create_question_prove(target)
                  
        else: # other predicates
            res_info = self.create_question_prove(target)
        
        if res_info is None:
            return [None] * 7
        (
            conclusion,
            add_cdls,
            add_conditions,
            target_value, 
            target_str,
            target_cdl
        ) = tuple([res_info[k] for k in res_info])
        
        for k, v in SYMBOL_MAPPING_2.items():
            conclusion = conclusion.replace(k, v)
            target_str = target_str.replace(k, v)
            add_conditions = [c.replace(k, v) for c in add_conditions]
            
        caption_str = ', '.join(conditions)

        if len(add_conditions) > 0:
            if problem_text.endswith(', '):
                problem_text += f"given {', '.join(add_conditions)}, {target_str}."
            elif problem_text == '':
                problem_text += f"Given {', '.join(add_conditions)}, {target_str}."
            else:
                problem_text += f". Given {', '.join(add_conditions)}, {target_str}."
        else:
            if problem_text.endswith(', '):
                problem_text += f"{target_str}."
            elif problem_text == '':
                problem_text += f"{target_str[0].upper() + target_str[1:]}"
            else:
                problem_text += f". {target_str[0].upper() + target_str[1:]}"
        if not problem_text.endswith('.'):
            problem_text += '.'
        return conclusion, add_cdls, add_conditions, target_value, target_cdl, problem_text, caption_str
    
    def target_tuple_to_clause(self, target):
        keys = list(self.predicate_GDL['Relation'].keys())
        template = [k for k in keys if target[0] in k][0]
        _, items = parse_clause(template)
        items = list(','.join(items))
        j = 0
        for i in range(len(target[1])):
            if items[j] == ',':
                j += 1
            items[j] = target[1][i]
            j += 1
        clause = f"{target[0]}({''.join(items)})"
        return clause
        
    def formulate(self, image_based=True, strict=False):
        start_time = time.time()
        self.solver.init_search(self.problem_CDL)
        self.solver.search()
        # self.solver.bfs_search()
        cost_time = time.time() - start_time
        if self.debug:
            print(f"Search took {cost_time:.5f} seconds. ")
        # construct condition graph
        condition_graph = ConditionGraph(self.solver.problem.condition.items)
        condition_graph.construct_graph()
        
        # find potential geo relation ()
        (
            target_type,
            target, 
            problem_level, 
            solution_str, 
            solution_dict,
            theorems,
            available_num
        ) = self.find_target_and_solution(condition_graph, strict=strict)

        if target is None:
            return None, None
        # create question and solution for this geo relation
        (
            conclusion,
            add_cdls,
            add_conditions,
            target_value, 
            target_cdl, 
            problem_text,
            cap_str
        ) = self.create_question(target)
        self.text_cdls += add_cdls
        self.image_cdls += add_cdls

        if image_based:
            solution_str = f'In the figure, we can get:\n{cap_str}.\n{solution_str}'
        else:
            problem_text += f'In the figure, {cap_str}.'
        
        if len(add_conditions) != 0:
            solution_str += f"\n<because> {', '.join(add_conditions)}, <therefore> {conclusion}."
        else:
            solution_str += f"\n<therefore> {conclusion}."

        info_dict_for_symbolic = {
            "problem_level": problem_level,
            "problem_text_en": problem_text,
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            "goal_cdl": target_cdl,
            "problem_answer": str(target_value),
            "theorems": theorems,
            "time": round(cost_time, 3),
            "available_targets_num": available_num
        }
        info_dict_for_llm = {
            "problem_type": target_type,
            "problem_level": problem_level,
            "problem_text": problem_text,
            "problem_answer": str(target_value),
            "solution_str": solution_str,
            "solution_dict": solution_dict
        }
            
        return info_dict_for_symbolic, info_dict_for_llm


def check_predicate_combs(pred_base_combs, pred_rel_combs):
    # filter all 'IsMidsegmentOfQuadrilateral' if pred_base not in 
    predicates_1 = ['Square', 'Rectangle', 'Rhombus', 'Parallelogram', 'Trapezoid', 'IsoscelesTrapezoid', 'RightTrapezoid', 'Kite']
    if 'IsMidsegmentOfQuadrilateral' in pred_rel_combs:
        if pred_base_combs[0] not in predicates_1:
            return False
        
    return True 

            
def build_input_args(pred_base_combs, 
                     pred_rel_combs, 
                     n_more_lines,
                     sampling_num):
    input_args = []
    for predicate_base in pred_base_combs:
        for predicate_rel in pred_rel_combs:
            for _ in range(sampling_num):
                res = check_predicate_combs(predicate_base, predicate_rel)
                if not res:
                    continue
                input_args.append(
                    (predicate_base, predicate_rel, n_more_lines, None)
                )
    return input_args

if __name__ == '__main__':
    setup_seed(96)
    
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    # for i in range(10):
        # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=19
    cnt = 0
    # total_len = len(list(itertools.product(PREDICATES_ENT, PREDICATES_REL)))
    # for clauses_base, clauses_rel in itertools.product(PREDICATES_ENT, PREDICATES_REL):
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=1)
    total_len = len(input_args_1)
    for predicate_base, predicate_rel, n_more_lines, color_config in input_args_1:
        # cnt += 1
        # if cnt < 167:
        #     continue

        # predicate_base = random.choices(PREDICATES_ENT, k=1)
        # predicate_rel = random.choices(PREDICATES_REL, k=1)
        predicate_base = [
            # "Square",
            # "RightTriangle"
            # "IsoscelesTriangle"
            "Parallelogram"
            
        ]
        predicate_rel = [
            # 'IsMidpointOfLine', 
            # 'IsTangentOfCircle',
            # "PerpendicularBetweenLine",
            "IsBisectorOfAngle"
            # 'IsDiameterOfCircle'
            # 'PerpendicularBetweenLine'
        ]
        cg = ClauseGenerator(predicate_GDL, theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            predicate_base, 
            predicate_rel, 
            n_more_lines=0
        )
        states = cg.states
        
        # states =  {'points': ['a', 'b', 'c', 'd', 'e'], 'lines': [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd'), ('c', 'e'), ('b', 'e')], 'circles': [], 'polygons': [('a', 'b', 'c', 'd'), ('b', 'c', 'e')], 'constraints': ['ParallelBetweenLine(ad,bc)', 'ParallelBetweenLine(ba,cd)', 'Equal(MeasureOfAngle(bce),MeasureOfAngle(ecd))'], 'constraints_base': ['ParallelBetweenLine(ad,bc)', 'ParallelBetweenLine(ba,cd)'], 'points_on_circle': {}}
        # c_cdls =  ['Shape(ab,bc,cd,da)', 'Shape(ce)']
        # t_cdls =  ['Parallelogram(abcd)', 'IsBisectorOfAngle(ce,bcd)'] 
        
        print(f'------- {cnt} / {total_len} Allocator Inputs -------')
        print('states = ', states)
        print('c_cdls = ', c_cdls)
        print('t_cdls = ', t_cdls)

        allocator = Allocator(states, c_cdls, t_cdls, replace_chars=False)
        allocator.allocate()
            
        print(f'------- {cnt} / {total_len} Formulated CDLs -------')
        print('Text CDLs: ')
        for t_cdl in allocator.formulated_cdls['text_cdls']:
            print('\t', t_cdl)
        print('Construct CDLs: ')
        for c_cdl in allocator.formulated_cdls['construct_cdls']:
            print('\t', c_cdl)
            
        # t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
        t_info = json.load(open("json/t_info_new.json"))
        t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
        t_freq_info = {k: t_info[k][-1] for k in t_names}
        
        plotter = Plotter(allocator.states,
                            allocator.formulated_cdls['text_cdls'],
                            allocator.formulated_cdls['construct_cdls'],
                            allocator.formulated_cdls['image_cdls'],
                            replace_characters=False)
        plotter.plot()
        plotter.save_fig('test', 'imgs_test')
        
        predicate_GDL_search = json.load(open('json/predicate_GDL_for_search.json', 'r'))
        theorem_GDL_search = json.load(open('json/theorem_GDL_for_search.json', 'r'))
        goal_finder = TargetFinder(
            predicate_GDL_search,
            theorem_GDL_search,
            t_info,
            t_freq_info,
            allocator.states, 
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls'],
            allocator.image_cdls,
            replace_characters=False,
            predicate_num=len(predicate_base) + len(predicate_rel),
            debug=True
        )
        info_dict_for_symbolic, info_dict_for_llm = goal_finder.formulate()

        print(f'---------- {cnt} / {total_len} Problem Text ----------')
        if info_dict_for_llm is not None:
            print(info_dict_for_llm['problem_text'])
        print(f'--------- {cnt} / {total_len} Forward Search ---------')
        if info_dict_for_llm is not None:
            print(info_dict_for_llm['solution_str'])
        print('==============================================')