import json
import random
import re
import string
from typing import Dict, Tuple

import numpy as np
from sympy import Eq, Symbol, solve, total_degree

from allocator import Allocator
from formalgeo.core import EquationKiller as EqKiller
from formalgeo.data import DatasetLoader
from formalgeo.parse import inverse_parse_one_theorem, parse_theorem_seqs
from formalgeo.problem.condition import Goal
from formalgeo.solver import BackwardSearcher, ForwardSearcher, Interactor
from generator import ClauseGenerator
from graph import (ConditionGraph, ConditionNode, display_solution, draw_graph,
                   topological_sort)
from plotter import Plotter
from solver import FormalGeoSolver, InterGPSSolver
from utils import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,
                   PRESET_COLOR_PROBS, PRESET_COLORS,
                   clause_to_nature_language, parse_clause, replace_for_clause,
                   setup_seed)


class TargetFinder(): 
    def __init__(self, 
                 dl,
                 t_info,
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
        self.predicate_GDL = dl.predicate_GDL
        self.theorem_GDL = dl.theorem_GDL
        self.problem_id = problem_id
        self.t_info = t_info
        self.debug = debug
        
        if replace_characters:
            self.replace_characters()

        assert solver_type in ['formalgeo', 'intergps']
        if solver_type == 'formalgeo':
            self.solver = FormalGeoSolver(
                dl.predicate_GDL,
                dl.theorem_GDL,
                strategy="beam_search",
                max_depth=20, 
                beam_size=6,
                t_info=t_info,
                debug=False
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
            "problem_answer": "1",
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
    
    def find_target(self, condition_graph: ConditionGraph):
        # find variable to solve, choose deepest conditoins
        pos_in_selection_tree = list(self.solver.leveled_condition.keys())
        end_pos = max(pos_in_selection_tree, key=len)
        # end_conditions = list(self.solver.leveled_condition[end_pos].values())
        end_conditions = self.solver.problem.condition.items[-10:]
        problem_level = len(end_pos)
        new_targets_cal = []
        new_targets_prv = []

        # condition: (predicate, expr/items, dependency, theorem, depth)
        # for potential calculation target: 
        # 1. predicate == 'Equation'
        # 2. only has <= 2 vars
        # 3. only has linear term
        # 4. can't be simple: line_addition, flat angle
        for condition in end_conditions:
            if condition[0] == 'Equation':
                if len(condition[1].free_symbols) <= 2:
                    if total_degree(condition[1]) <= 1:
                        if condition[3][0] not in ['line_addition', 'flat_angle']:
                            new_targets_cal.append(condition)
            # else:
            #     new_targets_prv.append(condition)
        
        new_targets = new_targets_cal + new_targets_prv
        
        theorems_for_targets = {}
        for target in new_targets:
            _, _, theorems = self.find_solution_for_target(condition_graph, target)
            theorems_for_targets[target] = theorems
        # random choice
        # new_targets =  random.choice(new_targets) 
        # depth first
        # depths = [condition_graph.calculate_depth(t) for t in new_targets]
        # target_idx = sorted(range(len(depths)), key=lambda i: depths[i], reverse=True)[0]
        # new_targets = new_targets[target_idx]
        # todo: modify problem level
        def diversity_idx(theorem_list):
            # 1. sort by len of theorems first
            # 2. sort by token diversity of theorems
            score_1 = len(theorem_list)
            token_set = set()
            for item in theorem_list:
                tokens = item.split('(')[0].split('_')
                token_set.update(tokens)
            score_2 = len(token_set)
            return (score_1, score_2)
        
        chosen_target = max(theorems_for_targets, key=lambda k: diversity_idx(theorems_for_targets[k]))
        problem_level = None
        for key, value in self.solver.leveled_condition.items():
            if chosen_target in list(value.values()):
                problem_level = len(key)
                break
            
        problem_level = len(theorems_for_targets[chosen_target]) \
            if problem_level == None else problem_level
        return chosen_target, problem_level
    
    def find_solution_for_target(self, 
                         condition_graph: ConditionGraph, 
                         target_condition: Tuple):
        sub_nodes, sub_nodes_adj_table = condition_graph.backward_construct_sub_graph([target_condition])
    
        # sort by index of applying theorem 
        sub_nodes = topological_sort(sub_nodes, sub_nodes_adj_table)

        solutions = []
        solution_dict = {}
        theorems_formal = []
        
        for i, node in enumerate(sub_nodes):
            theorem = node.value[3][0]
            if theorem not in ['prerequisite', 'extended', 'solve_eq']:
                theorems_formal.append(
                    inverse_parse_one_theorem(node.value[3], self.solver.parsed_theorem_GDL)
                )
            statement = str(node).split('|')[0].strip()
            if 'Equation' in statement:
                pattern = r"Equation\((.*?)\)"
                matches = re.findall(pattern, statement)[0]
                statement = f"{matches} = 0"
            if node.value[-1] == 0 or theorem in ['angle_addition', 'line_addition']:
                solution_i = f"{theorem}: {statement} [{i}]"
                pre_idx = [-1]
            else:
                pre_nodes = [condition_graph.nodes[j] for j in node.value[2]]
                pre_idx = [sub_nodes.index(n) for n in pre_nodes]
                pre_idx_str = "".join([f"[{x}]" for x in pre_idx])
                solution_i = f"{theorem}: {pre_idx_str} => {statement} {[i]}"
            solutions.append(solution_i)
            solution_dict[i] = {
                "theorem": theorem,
                "condition": str(pre_idx),
                "statement": statement
            }
            
        solution_str = "Solution: \n" + "\n".join(solutions)
        return solution_dict, solution_str, theorems_formal
    
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
            self.problem_CDL['text_cdl'] + self.problem_CDL['image_cdl'],
            self.natural_template)

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
                    ],
                    add_conditions = [
                        f"angle {angle_1} = {str(expr_1)}",
                        f"angle {angle_2} = {str(expr_2)}",
                    ]
                    conclusion = f"{new_sym} = {target_value}"
                    
            else: # len(free_symbols) == 1
                sym = list(target[1].free_symbols)[0]
                target_value = solve(Eq(target[1], 0))[0]
                conclusion = f"{sym} = {target_value}"
                if 'll' in str(sym):
                    target_str = f"Find length of {str(sym).split('ll_')[0].upper()}"
                    target_cdl = f"Value(LengthOfLine({str(new_sym)}))"
                else:
                    target_str = f"Find measure of angle {str(sym).split('ma_')[0].upper()}"
                    target_cdl = f"Value(MeasureOfAngle({str(new_sym)}))"
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
        info_dict = display_solution(condition_graph, target, interact_solver.problem.goal)
        
      
        return info_dict
        
    def formulate(self):
        self.solver.init_search(self.problem_CDL)
        self.solver.search()
        # if self.debug:
        #     for condition in self.solver.problem.condition.items:
        #         print(condition)
            
        condition_graph = ConditionGraph(self.solver.problem.condition.items)
        condition_graph.construct_graph()
        # find potential geo relation ()
        target, problem_level = self.find_target(condition_graph)
        # find solution for this geo relation
        solution_dict, solution_str, theorems = self.find_solution_for_target(condition_graph, target)
        # create question and solution for this geo relation
        target_value, target_cdl, text, add_conditions, add_cdls, conclusion = self.create_question(target)
        self.text_cdls += add_cdls
        self.image_cdls += add_cdls
        
        if len(add_conditions) != 0:
            solution_str += f"\n{', '.join(add_conditions)} => {conclusion}."
        else:
            solution_str += f"Therefore, {conclusion}."
        
        # if self.debug:
        #     draw_graph(condition_graph, 
        #         idx="test", 
        #         target_condition=target,
        #         img_dir="imgs_test")
            
        
        new_problem_CDL = {
            "problem_id": self.problem_CDL['problem_id'],
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            "goal_cdl": target_cdl,
            "theorem_seqs": theorems,
            "problem_answer": str(target_value)
        }
        
        new_solution_info = self.re_solve_for_target(new_problem_CDL)
        
        if self.debug:
            print('------------- Forward Search -------------')
            print(solution_str)
            print('------------- Interactive -------------')
            print(new_solution_info['solution_str'])
        
        info_dict_for_symbolic = {
            "problem_id": "",
            "problem_img": "",
            "problem_level": problem_level,
            "problem_text_en": text,
            "construction_cdl": self.constr_cdls,
            "text_cdl": self.text_cdls,
            "image_cdl": self.image_cdls,
            "goal_cdl": target_cdl,
            "problem_answer": target_value,
        },
        info_dict_for_llm = {
            "problem_id": "",
            "problem_img": "",
            "problem_level": problem_level,
            "problem_text_en": text,
            "problem_descrition": "",
            "problem_answer": target_value,
            "solution_str": solution_str,
            "solutions": solution_dict
        }
            
        return info_dict_for_symbolic, info_dict_for_llm
            
            
        
if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(10):
        # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
        clauses_base = random.choices(PREDICATES_ENT, k=1)
        clauses_rel = random.choices(PREDICATES_REL, k=2)
        # clauses_base = [
        #     "RightTrapezoid",
        # ]
        # clauses_rel = [
        #     'IsBisectorOfAngle', 
        #     # 'IsMidsegmentOfTriangle',
        #     # 'IsAltitudeOfQuadrilateral',
        #     # 'IsIncenterOfTriangle',
        #     # "IsAltitudeOfTriangle",
        #     # "IsCircumcenterOfQuadrilateral",
        #     # "IsMidpointOfArc"
        #     ]
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_more_lines=0
        )
        states = cg.states
        
        # states = {'points': ['a', 'b', 'c', 'd', 'e'], 'lines': [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd'), ('d', 'e'), ('a', 'c')], 'circles': [], 'polygons': [('a', 'b', 'c', 'd'), ('a', 'b', 'c'), ('a', 'c', 'd')], 'constraints': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)', 'Equal(MeasureOfAngle(ade),MeasureOfAngle(edc))'], 'constraints_base': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)'], 'points_on_circle': {}}
        # c_cdls = ['Shape(ab,bc,cd,da)', 'Shape(de)']
        # t_cdls = ['RightTrapezoid(abcd)', 'IsBisectorOfAngle(de,adc)']  
        
        print('---------- Allocator Inputs ----------')
        print(states)
        print('c_cdls: ', c_cdls)
        print('t_cdls: ', t_cdls)

        allocator = Allocator(states, c_cdls, t_cdls)
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
        
        plotter = Plotter(allocator.states,
                            allocator.formulated_cdls['text_cdls'],
                            allocator.formulated_cdls['construct_cdls'],
                            allocator.formulated_cdls['image_cdls'],
                            replace_characters=False)
        plotter.plot()
        plotter.save_fig('test', 'imgs_test')

        goal_finder = TargetFinder(
            dl,
            t_info,
            allocator.states, 
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls'],
            allocator.image_cdls,
            replace_characters=False,
            solver_type='intergps',
            debug=True
        )
        info_dict_for_symbolic, info_dict_for_llm = goal_finder.formulate()
        print('---------- Target and Solution ----------')
        print(info_dict_for_llm['problem_text_en'])
        print(info_dict_for_llm['solution_str'])

        print('==============================================')