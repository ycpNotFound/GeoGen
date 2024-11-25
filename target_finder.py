import json
import random
import re
import string
from typing import Dict, Tuple
from collections import deque

import numpy as np
from sympy import Eq, Symbol, solve, total_degree, Integer, Float, sqrt

from allocator import Allocator
from formalgeo.core import EquationKiller as EqKiller
from formalgeo.data import DatasetLoader
from formalgeo.parse import inverse_parse_one_theorem, parse_theorem_seqs, inverse_parse_one
from formalgeo.problem.condition import Goal
from formalgeo.solver import BackwardSearcher, ForwardSearcher, Interactor
from formalgeo.problem import Problem
from generator import ClauseGenerator
from graph import (ConditionGraph, ConditionNode, display_solution, draw_graph,
                   topological_sort, topological_sort_bfs)
from plotter import Plotter
from solver import FormalGeoSolver, InterGPSSolver
from utils import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2, 
                   PRESET_COLOR_PROBS, PRESET_COLORS,
                   clause_to_nature_language, parse_clause, replace_for_clause,
                   setup_seed, SYMBOL_MAPPING_2)


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
                 debug=False):
        self.p_pos = allocater_states['p_pos']
        self.lines = allocater_states['lines']
        self.circles = allocater_states['circles']
        self.points_on_circle = allocater_states['points_on_circle']
        # self.clauses = allocater_states['clauses_base'] + allocater_states['clauses']
        self.clauses = allocater_states['clauses']
        
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

        assert solver_type in ['formalgeo', 'intergps']
        if solver_type == 'formalgeo':
            self.solver = FormalGeoSolver(
                predicate_GDL,
                theorem_GDL,
                strategy="beam_search",
                max_depth=5, 
                beam_size=6,
                t_info=t_info,
                t_freq_info=t_freq_info,
                # debug=debug
            )
        elif solver_type == 'intergps':
            self.solver = InterGPSSolver(
                debug=debug
            )
        self.problem_CDL = {
            "problem_id": problem_id,
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            "goal_cdl": f"Value(LengthOfLine({''.join(self.lines[0])}))",
            # "goal_cdl": f"Value(MeasureOfAngle(bed))",
            "problem_answer": "45",
            "point_positions": self.p_pos,
            "line_instances": self.lines,
            "circle_instances": self.circles
        }
        natural_template_path = "json/natural_language_template.json"
        self.natural_template = json.load(open(natural_template_path, 'r'))
        self.symbols = []
        
    @property
    def states(self):
        return {
            "p_pos": self.p_pos,
            "lines": self.lines,
            "circles": self.circles,
            "points_on_circle": self.points_on_circle,
            "clauses": self.clauses
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
        self.points_on_circle = {mapping[k]: [mapping[x] for x in v] 
                                 for k, v in self.points_on_circle.items()}
        # clauses
        self.clauses = [replace_for_clause(c, mapping) for c in self.clauses]
        self.text_cdls = [replace_for_clause(c, mapping) for c in self.text_cdls]
        self.constr_cdls = [replace_for_clause(c, mapping) for c in self.constr_cdls]
        self.image_cdls = [replace_for_clause(c, mapping)for c in self.image_cdls]
        return
    
    def find_target_and_solution(self, condition_graph: ConditionGraph):
        # find variable to solve, 
        max_depth = max([len(k) for k in self.solver.leveled_condition])

        conditions_to_smaple = []
        for k, v in self.solver.leveled_condition.items():
            if len(k) == max_depth:
                conditions_to_smaple += list(v.values())
            if max_depth > 1 and len(k) == max_depth - 1:
                conditions_to_smaple += list(v.values())

        if len(conditions_to_smaple) == 1:
            for k, v in self.solver.leveled_condition.items():
                if len(k) == max_depth-1:
                    conditions_to_smaple += list(v.values())
        new_targets_cal, new_targets_prv = [], []

        # the first filter: 
        # for potential calculation target: 
        # 1. only has <= 2 vars
        # 2. only has linear term, degree <= 1
        # 3. if has 2 vars, can not be both solved value
        # 4. only has symbols begin with 'll_' or 'ma_'
        for condition in conditions_to_smaple:
            if condition[0] == 'Equation':
                f1 = len(condition[1].free_symbols) <= 2
                f2 = condition[1].as_poly(*list(condition[1].free_symbols)).total_degree() <= 1
                syms = [str(x) for x in list(condition[1].free_symbols)]
                f3 = True
                if len(condition[1].free_symbols) == 2:
                    a = 1
                    if all([
                        self.solver.problem.condition.value_of_sym[sym] is not None 
                        for sym in list(condition[1].free_symbols)
                    ]):
                        f3 = False
                        
                f4 = all(['ll_' in sym or 'ma_' in sym for sym in syms])
                if all([f1, f2, f3, f4]):
                    new_targets_cal.append(condition)
            else:
                if condition[0] in PREDICATES_REL + PREDICATES_ENT:
                    new_targets_prv.append(condition)
                    
        # the second filter: 
        # control the number angle target, line target and prove target
        # so that the distribution of type won't be too sharp
        angle_targets = [t for t in new_targets_cal if 'ma_' in str(t[1])]
        line_targets = [t for t in new_targets_cal if 'll_' in str(t[1])]
        new_targets_cal = angle_targets[:5] + line_targets[:5]
        new_targets = new_targets_cal + new_targets_prv[:5]
        
        # find solution / theorems for each target
        theorems_for_targets = {}
        solution_for_targets = {}
        level_for_targets = {}
        for target in new_targets:
            (
                solution_str, 
                theorems, 
                sub_nodes
            ) = self.find_solution_for_target(
                self.solver.problem,
                condition_graph, 
                target, 
            )
            level = max_depth
            for key, value in self.solver.leveled_condition.items():
                if target in list(value.values()):
                    level = len(key)
                    break
            theorems_for_targets[target] = theorems
            solution_for_targets[target] = solution_str
            level_for_targets[target] = level

        if self.debug:
            solution_for_targets_ = {
                str(k): v.split('\n') for k, v in
                solution_for_targets.items()
            }
            with open('json/solution_test.json', 'w', encoding='utf-8') as f:
                json.dump(solution_for_targets_, f, indent=4, ensure_ascii=False)
        
        # the third filter: 
        # 1. sort by len of theorems (more but not too large)
        # 2. sort by num of unsolved symbols (less)
        # 3. sort by token diversity of theorems (more)
        def filter_idx(target, theorem_list, level):
            score_1 = len(theorem_list) 
            if len(theorem_list) >= level + 2:
                score_1 = 0
                
            score_2 = 0
            if target[0] == 'Equation':
                score_2 = - len(target[1].free_symbols)
                if abs(target[1].as_coefficients_dict().get(1, 0)) > 180: # delete some angle > 180 to be target
                    score_2 = -100
                
            token_set = set()
            for item in theorem_list:
                tokens = item.split('(')[0].split('_')
                token_set.update(tokens)
            score_3 = len(token_set)
            return (score_1, score_2, score_3)
        
        # reserve top-5 targets and random choice
        chosen_targets = sorted(
            theorems_for_targets, 
            key=lambda k: filter_idx(
                k, 
                theorems_for_targets[k],
                level_for_targets[k]
            ), 
            reverse=True
        )[:5]
        if len(chosen_targets) == 0:
            return None, None, None, None
        chosen_target = random.choice(chosen_targets)
        chosen_thoerems = theorems_for_targets[chosen_target]
        chosen_solution = solution_for_targets[chosen_target]
        problem_level = len(theorems_for_targets[chosen_target]) 
        return chosen_target, problem_level, chosen_solution, chosen_thoerems
    
    def find_solution_for_target(
            self, 
            problem: Problem,
            condition_graph: ConditionGraph, 
            target_condition: Tuple
        ):
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
            self.natural_template,
            upper=False,
            symbol2nature=False
        )
        pred_ignore = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle']
        extend_nodes = [n for n in sub_nodes if n.value[3][0] == 'extended' and n.value[0] not in pred_ignore]

        theorems_formal = []
        solution_str = "Solution: "
        step_count = 0
        
        # special token: 
        # <by> - theorem
        # <because> - parent condition
        # <therefore> - extend condition
        
        for i, node in enumerate(sub_nodes):
            theorem = node.value[3][0]
            statement = sub_nodes_statements[i]
            
            if theorem not in ['prerequisite', 'extended', 'solve_eq']:
                theorems_formal.append(
                    inverse_parse_one_theorem(node.value[3], self.solver.parsed_theorem_GDL)
                )
                
            if theorem == 'prerequisite': 
                # add all extended condition
                extend_conditions = []
                queue = deque([n for n in extend_nodes if node.idx in n.value[2]])
                while queue:
                    extend_node = queue.popleft()
                    extend_idx = sub_nodes.index(extend_node)
                    extend_statement = sub_nodes_statements[extend_idx]
                    extend_conditions.append(extend_statement)
                    
                    for n in extend_nodes:
                        if extend_node.idx in n.value[2]:
                            queue.append(n)
                if len(extend_conditions) != 0:
                    step_count += 1
                    solution_str += f"\nStep {step_count}: <because> {statement}, <therefore> {', '.join(extend_conditions)}. "

                    
            elif theorem == 'extended': 
                # add extended condition in 'prerequisite' or other theorems
                pass
                
            else: # using theorem 
                step_count += 1
                solution_str += f'\nStep {step_count}: <by> {theorem}, '
                
                # add all parent conditions
                parent_conditions = []
                for parent_idx in node.value[2]:
                    if parent_idx not in sub_nodes_idx:
                        continue
                    # pass condition like Line(AB)
                    parent_node = sub_nodes[sub_nodes_idx.index(parent_idx)]
                    if parent_node.value[0] in pred_ignore:
                        continue
                    
                    parent_statement = sub_nodes_statements[sub_nodes_idx.index(parent_idx)]
                    parent_conditions.append(parent_statement)

                if len(parent_conditions) != 0:
                    solution_str += '<because> '
                    solution_str += ', '.join(parent_conditions) + ', '

                extend_conditions = [statement]
                # add all extended conditions
                queue = deque([n for n in extend_nodes if node.idx in n.value[2]])
                while queue:
                    extend_node = queue.popleft()
                    extend_idx = sub_nodes.index(extend_node)
                    extend_statement = sub_nodes_statements[extend_idx]
                    extend_conditions.append(extend_statement)
                    
                    for n in extend_nodes:
                        if extend_node.idx in n.value[2]:
                            queue.append(n)
                solution_str += f"<therefore> {', '.join(extend_conditions)}. "
            
        return solution_str, theorems_formal, sub_nodes
    
    def create_question(self, target: Tuple):
        # create target and added conditions
        text = random.choice([
            "In this figure, ",
            "As shown in the figure, ",
            "In the given diagram, ",
            "According to the shown figure, ",
            "Based on the figure provided, "
        ])

        conditions = clause_to_nature_language(
            # self.problem_CDL['text_cdl'] + self.problem_CDL['image_cdl'],
            self.problem_CDL['text_cdl'] + self.problem_CDL['construction_cdl'],
            self.natural_template
        )

        # information to return
        conclusion = None
        add_cdls, add_conditions = [], []
        target_value, target_str, target_cdl = None, None, None
        
        # - ll_cd + ll_ed, ma_abc + ma_edf - 180
        if target[0] == 'Equation': 
            if len(target[1].free_symbols) == 2:
                # assign value to one, and solve the other
                target_sym = random.choice(list(target[1].free_symbols))
                another_sym = list(target[1].free_symbols - set([target_sym]))[0]
                if 'll' in str(target_sym):
                    target_line = str(target_sym).split('_')[-1].upper()
                    other_line = str(another_sym).split('_')[-1].upper()
                    value = random.randint(1, 10)
                    expr = target[1].subs({another_sym: value})
                    
                    target_value = solve(Eq(expr, 0))[0]
                    target_str = f"Find the length of {target_line}"
                    target_cdl = f"Value(LengthOfLine({target_line}))"
                    add_cdls = [f"Equal(LengthOfLine({other_line}),{value})"]
                    add_conditions = [f"{other_line} = {value}"]
                    conclusion = f"{target_line} = {target_value}"
                    
                elif 'ma' in str(target_sym):
                    syms = list(target[1].free_symbols)
                    angle_1 = str(syms[0]).split('_')[-1].upper()
                    angle_2 = str(syms[1]).split('_')[-1].upper()
                    # find angle measure first
                    angle_1_val = self.solver.problem.condition.value_of_sym[syms[0]]
                    angle_2_val = self.solver.problem.condition.value_of_sym[syms[1]]
                    flag_1 = type(angle_1_val) in [Integer, Float]
                    flag_2 = type(angle_2_val) in [Integer, Float]
                    
                    if flag_2:
                        target_value = angle_2_val
                        target_str = f"Find measure of \\angle {angle_2}"
                        target_cdl = f"Value(MeasureOfAngle({angle_2}))"
                        add_conditions = []
                        add_cdls = []
                        conclusion = f"\\angle {angle_2} = {target_value}"
                    elif flag_1:
                        target_value = angle_1_val
                        target_str = f"Find measure of \\angle {angle_1}"
                        target_cdl = f"Value(MeasureOfAngle({angle_1}))"
                        add_conditions = []
                        add_cdls = []
                        conclusion = f"\\angle {angle_1} = {target_value}"
                    else:
                        # avoid to directly assign value to angle
                        v1 = random.randint(1, 10)
                        v2 = random.randint(1, 10)
                        v3 = random.randint(1, 10)
                        v4 = random.randint(1, 10)
                        new_sym = self.add_new_symbol()
                        expr_1 = v1*new_sym + v2
                        expr_2 = v3*new_sym + v4
                        expr = target[1].subs({syms[0]: expr_1,
                                            syms[1]: expr_2})
                        
                        target_value = solve(Eq(expr, 0))[0]
                        target_str = f"Find value of {str(new_sym)}"
                        target_cdl = f"Value({str(new_sym)})"
                        add_cdls = [
                            f"Equal(MeasureOfAngle({angle_1}),{str(expr_1)})",
                            f"Equal(MeasureOfAngle({angle_2}),{str(expr_2)})",
                        ]
                        add_conditions = [
                            f"\\angle {angle_1} = {str(expr_1)}",
                            f"\\angle {angle_2} = {str(expr_2)}",
                        ]
                        conclusion = f"{new_sym} = {target_value}"
                    
            else: # len(free_symbols) == 1
                sym = list(target[1].free_symbols)[0]
                angle = str(sym).split('_')[-1].upper()
                target_value = solve(Eq(target[1], 0))[0]
                conclusion = f"\\angle {angle} = {target_value}"
                if 'll' in str(sym):
                    target_str = f"Find length of {str(sym).split('ll_')[-1].upper()}"
                    target_cdl = f"Value(LengthOfLine({str(sym)}))"
                else:
                    target_str = f"Find measure of angle {str(sym).split('ma_')[-1].upper()}"
                    target_cdl = f"Value(MeasureOfAngle({str(sym)}))"
            
                  
        else: # other predicates
            target_value = str(target[1])
            if target[0] in PREDICATES_ENT:
                conclusion = self.natural_template[target[0]][0].format(points=''.join(target[1]))
                clause = f"{target[0]}({''.join(target[1])})"
            else:
                clause = self.target_tuple_to_clause(target)
                _, items = parse_clause(clause)
                conclusion = self.natural_template[target[0]][0].format(
                    p1=items[0], p2=items[1])
            
            target_str = f'Prove that {conclusion}'
            target_cdl = f"Relation({clause})"
            
        for k, v in SYMBOL_MAPPING_2.items():
            conclusion = conclusion.replace(k, v)
            target_str = target_str.replace(k, v)
            add_conditions = [c.replace(k, v) for c in add_conditions]
            
        conditions += add_conditions
        text += ', '.join(conditions)
        text += f". {target_str}."
        return target_value, target_cdl, text, add_conditions, add_cdls, conclusion
    
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
    
    def re_solve_for_target(self, problem_CDL):
        interact_solver = Interactor(self.predicate_GDL, self.theorem_GDL, self.t_info,
                        debug=False
                        )
        interact_solver.load_problem(problem_CDL)

        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            interact_solver.apply_theorem(t_name, t_branch, t_para)
        
        interact_solver.problem.check_goal()
        
        # construct graph
        condition_graph = ConditionGraph(interact_solver.problem.condition.items)
        condition_graph.construct_graph()
        target = interact_solver.problem.condition.items[-1]
        solution, _, __ = self.find_solution_for_target(
            interact_solver.problem,
            condition_graph, 
            target
        )
        return solution
        
    def formulate(self):
        self.solver.init_search(self.problem_CDL)
        self.solver.search()
        # self.solver.bfs_search()
        
        # construct condition graph
        condition_graph = ConditionGraph(self.solver.problem.condition.items)
        condition_graph.construct_graph()
        
        # find potential geo relation ()
        (
            target, 
            problem_level, 
            solution_str, 
            theorems
        ) = self.find_target_and_solution(condition_graph)

        if target is None:
            return None, None
        # create question and solution for this geo relation
        (
            target_value, 
            target_cdl, 
            text, 
            add_conditions, 
            add_cdls, 
            conclusion
        ) = self.create_question(target)
        self.text_cdls += add_cdls
        self.image_cdls += add_cdls

        if len(add_conditions) != 0:
            solution_str += f"\n<because> {', '.join(add_conditions)}, <therefore> {conclusion}."
        else:
            solution_str += f"\n<therefore> {conclusion}."
        
        # if self.debug:
        #     draw_graph(condition_graph, 
        #         idx="test", 
        #         target_condition=target,
        #         img_dir="imgs_test")
            
        
        
        if self.debug:
            print('-------------- Problem Text --------------')
            print(text)
            print('------------- Forward Search -------------')
            print(solution_str)
        
        info_dict_for_symbolic = {
            "problem_level": problem_level,
            "problem_text_en": text,
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            "goal_cdl": target_cdl,
            "problem_answer": target_value,
        }
        info_dict_for_llm = {
            "problem_level": problem_level,
            "problem_text": text,
            "problem_answer": target_value,
            "solution_str": solution_str,
        }
            
        return info_dict_for_symbolic, info_dict_for_llm
            
            
        
if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(10):
        # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
        clauses_base = random.choices(PREDICATES_ENT, k=1)
        clauses_rel = random.choices(PREDICATES_REL, k=1)
        clauses_base = [
            "Triangle",
        ]
        clauses_rel = [
            'IsPerpendicularBisectorOfLine', 
            # 'IsMidsegmentOfTriangle',
            # 'IsAltitudeOfQuadrilateral',
            # 'IsIncenterOfTriangle',
            # "IsAltitudeOfTriangle",
            # "IsCircumcenterOfQuadrilateral",
            # "IsMidpointOfArc"
            ]
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_more_lines=1
        )
        states = cg.states
        
        # states = {'points': ['a', 'b', 'c', 'd', 'e'], 'lines': [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd'), ('d', 'e'), ('a', 'c')], 'circles': [], 'polygons': [('a', 'b', 'c', 'd'), ('a', 'b', 'c'), ('a', 'c', 'd')], 'constraints': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)', 'Equal(MeasureOfAngle(ade),MeasureOfAngle(edc))'], 'constraints_base': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)'], 'points_on_circle': {}}
        # c_cdls = ['Shape(ab,bc,cd,da)', 'Shape(de)']
        # t_cdls = ['RightTrapezoid(abcd)', 'IsBisectorOfAngle(de,adc)']  
        
        print('---------- Allocator Inputs ----------')
        print(states)
        print('c_cdls: ', c_cdls)
        print('t_cdls: ', t_cdls)

        allocator = Allocator(states, c_cdls, t_cdls, replace_chars=True)
        allocator.allocate()
        print("---------- Location ----------")
        for p, pos in allocator.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
            
        print('---------- Formulated CDLs ----------')
        print('Text CDLs: ')
        for t_cdl in allocator.formulated_cdls['text_cdls']:
            print('\t', t_cdl)
        print('Construct CDLs: ')
        for c_cdl in allocator.formulated_cdls['construct_cdls']:
            print('\t', c_cdl)
            
        
        t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
        t_freq_info = json.load(open("json/theorem_freq.json", 'r', encoding='utf-8'))
        
        plotter = Plotter(allocator.states,
                            allocator.formulated_cdls['text_cdls'],
                            allocator.formulated_cdls['construct_cdls'],
                            allocator.formulated_cdls['image_cdls'],
                            replace_characters=False)
        plotter.plot()
        plotter.save_fig('test', 'imgs_test')
        
        predicate_GDL = json.load(open('json/predicate_GDL_for_search.json', 'r'))
        theorem_GDL = json.load(open('json/theorem_GDL.json', 'r'))
        goal_finder = TargetFinder(
            predicate_GDL,
            theorem_GDL,
            t_info,
            t_freq_info,
            allocator.states, 
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls'],
            allocator.image_cdls,
            replace_characters=False,
            # solver_type='intergps',
            solver_type='formalgeo',
            debug=True
        )
        info_dict_for_symbolic, info_dict_for_llm = goal_finder.formulate()

        print('==============================================')