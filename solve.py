import json
import random
import re
from typing import Dict, Optional, Tuple

import numpy as np

import sympy
from sympy import Eq
from tqdm import tqdm

from formalgeo.core import EquationKiller as EqKiller
from formalgeo.data import DatasetLoader
from formalgeo.parse import parse_theorem_seqs
from formalgeo.problem.condition import Goal
from formalgeo.solver import Interactor
from graph import ConditionGraph, ConditionNode, display_solution, draw_graph
from utils.symbolic import move_subtractions_to_rhs


def construct_goal_from_condition(condition: Tuple):
    # example: 
    # answer: 15 | 0
    # item: y | ma_edg + ma_tcb - 180
    # premise: (condition_ids)
    # solved: True
    # solved_answer: 15 | 0
    # theorem: ('solve_eq', None, None)
    # type: 'algebra'
    goal = None
    if condition[0] == 'Equation':
        eq = Eq(condition[1], 0)
        # move const to right
        eq = Eq(eq.lhs - eq.lhs.as_coeff_Add()[0], eq.rhs - eq.lhs.as_coeff_Add()[0])
        
        goal = Goal()
        goal.answer = eq.rhs
        goal.item = eq.lhs
        goal.premise = condition[2]
        goal.solved = True
        goal.solved_answer = eq.rhs
        goal.theorem = condition[3]
        goal.type = 'algebra'
        
    else:
        pass

    return goal



def find_new_targets(solver: Interactor, 
                     condition_graph: ConditionGraph,
                     target_condition: Optional[Tuple]=None,
                     expand_num: int = 2,
                     sample_mode: str = "depth_first"):
    assert sample_mode in ['random', 'depth_first']
    target_sym = target_condition[1].free_symbols
    target_sym = list(target_sym)[0]
    value_of_sym = solver.problem.condition.value_of_sym

    # filter empty symbols and target symbol
    value_of_sym = {k: v for k, v in value_of_sym.items() \
        if v != None and k != target_sym}
    
    end_conditions = condition_graph.get_end_conditions()
    if target_condition and target_condition in end_conditions:
        end_conditions.remove(target_condition)
    
    new_targets_cal = []
    new_goals_cal = []

    for cond in end_conditions[::-1]:
        if cond[0] == 'Equation':
            if cond[3][0] not in ['line_addition', 'flat_angle']:
                new_targets_cal.append(cond)
                new_goals_cal.append(construct_goal_from_condition(cond))
        else:
            continue
    
    new_targets_prv = []
    new_goals_prv = []
    # todo: add prove rules and expand proving problems
    new_targets = new_targets_cal + new_targets_prv
    new_goals = new_goals_cal + new_goals_prv
    
    # random choice
    if sample_mode == 'random':
        random_idx = random.sample(list(range(len(new_targets))), expand_num)
        new_targets = [new_targets[i] for i in random_idx ]
        new_goals = [new_goals[i] for i in random_idx ]
    
    # depth first
    elif sample_mode == 'depth_first':
        depths = [condition_graph.calculate_depth(t) for t in new_targets]
        sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i], reverse=True)[:expand_num]
        new_targets = [new_targets[i] for i in sorted_indices]
        new_goals = [new_goals[i] for i in sorted_indices ]
        
    return new_targets, new_goals


def expr_to_upper(expr_str):
    if type(expr_str) != str:
        expr_str = str(expr_str).strip()
        
    def to_upper(match):
        return match.group().upper()
    
    expr_upper = expr_str
    pattern_1 = r'll_([a-zA-Z]+)'
    pattern_2 = r'ma_([a-zA-Z]+)'
    pattern_3 = r'la_([a-zA-Z]+)'
    pattern_4 = r'mar_([a-zA-Z]+)'
    pattern_5 = r'ht_([a-zA-Z]+)'

    # 使用 re.sub 替换所有匹配项
    expr_upper = re.sub(pattern_1, lambda m: m.group(1).upper(), expr_upper)
    expr_upper = re.sub(pattern_2, lambda m: '\\angle ' + m.group(1).upper(), expr_upper)
    expr_upper = re.sub(pattern_3, lambda m: '\\arc ' + m.group(1).upper(), expr_upper)
    expr_upper = re.sub(pattern_4, lambda m: '\\arc ' + m.group(1).upper(), expr_upper)
    expr_upper = re.sub(pattern_5, lambda m: 'height of triangle ' + m.group(1).upper(), expr_upper)
    
    if '=' not in expr_str and ' ' not in expr_str:
        if 'll_' in expr_str:
            expr_upper = f'the length of line {expr_upper}'
        elif 'ma_' in expr_str or 'mar_' in expr_str:
            expr_upper = f'the measure of {expr_upper}'
        elif 'la_' in expr_str:
            expr_upper = f'the length of {expr_upper}'
        elif 'ht_' in expr_str:
            expr_upper = f'the {expr_upper}' 
    return expr_upper

def create_problem_text(problem_CDL: Dict, goal: Goal):
    origin_p_text = problem_CDL["problem_text_en"]
    if 'Find' in origin_p_text:
        origin_conditions = origin_p_text.split('Find')[0]
    elif 'Prove' in origin_p_text:
        origin_conditions = origin_p_text.split('Prove')[0]
    
    if goal.answer == 0: # ll_ab - ll_bc
        eq_str = move_subtractions_to_rhs(Eq(goal.item, 0))
        eq_str = expr_to_upper(eq_str.strip())
        p_text = f"Prove that {eq_str}. "
    else:
        goal_str = expr_to_upper(goal.item)
        p_text = f"Find {goal_str}."

    problem_text = origin_conditions + p_text
    return problem_text
    

def solve_test():

    # problem_idx = 6981
    # problem_idx = 184
    # problem_idx = 1
    problem_idx = 1940
    # problem_idx = "test"
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL, t_info,
                        # debug=True
                        )
    
    problem_CDL = dl.get_problem(pid=problem_idx)
    
    solver.load_problem(problem_CDL)

    for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
        solver.apply_theorem(t_name, t_branch, t_para)
    
    solver.problem.check_goal()
    
    # save target_condition first
    target = solver.problem.condition.items[-1]
    
    # expand new conditions
    solver.expand_conditions()
    
    # construct graph
    condition_graph = ConditionGraph(solver.problem.condition.items)
    condition_graph.construct_graph()
    
    # draw condition DAG and display solution
    draw_graph(condition_graph, 
               idx="test", 
               target_condition=target,
               img_dir="imgs_test")
    draw_graph(condition_graph, 
               idx="test", 
               target_condition=None,
               img_dir="imgs_test")
    info_dict = display_solution(condition_graph, target, solver.problem.goal)
    print(info_dict['solution_str'])
    print('--------------------------')
    
    # expand new target and solution
    new_targets, new_goals = find_new_targets(solver, condition_graph, target)
    for new_target, new_goal in zip(new_targets, new_goals):
        draw_graph(condition_graph, 
               idx="test",
               target_condition=new_target)
        
        # construct new goal ?
        
        info_dict = display_solution(condition_graph, new_target, new_goal)
        print(info_dict['solution_str'])
        print('--------------------------')
    

def solve_and_expand_iteration():
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())
    save_path_1 = f"datasets/processed_data/fgo_{split}_solution.json"
    save_path_2 = f"datasets/processed_data/fgo_{split}_solution_expand.json"
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    solution_dict = {}
    solution_expand_dict = {}
    
    for problem_idx in tqdm(keys):
        problem_idx = int(problem_idx)
        solver = Interactor(dl.predicate_GDL, dl.theorem_GDL, t_info)
        
        problem_CDL = dl.get_problem(pid=problem_idx)
        solver.load_problem(problem_CDL)

        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            solver.apply_theorem(t_name, t_branch, t_para)
            
        solver.problem.check_goal()
        
        # save target_condition first
        target = solver.problem.condition.items[-1]
        
        # expand new conditions
        solver.expand_conditions()
        
        # construct graph
        condition_graph = ConditionGraph(solver.problem.condition.items)
        condition_graph.construct_graph()
        
        info_dict = display_solution(condition_graph, target, solver.problem.goal)
        info_dict["problem_text"] = problem_CDL["problem_text_en"]
        solution_dict[problem_idx] = info_dict
        
        # expand new target and solution
        new_targets, new_goals = find_new_targets(solver, condition_graph, target)
        for i in range(len(new_targets)):
            new_target, new_goal = new_targets[i], new_goals[i]
            info_dict_expand = display_solution(condition_graph, new_target, new_goal)
            # save the new problem text
            info_dict_expand["problem_text"] = create_problem_text(problem_CDL, new_goal)
            solution_expand_dict[f"{problem_idx}_{i}"] = info_dict_expand
        
        if problem_idx % 100 == 0:
            with open(save_path_1, 'w', encoding='utf-8') as f:
                f.write(json.dumps(solution_dict, indent=4, ensure_ascii=False))
            with open(save_path_2, 'w', encoding='utf-8') as f:
                f.write(json.dumps(solution_expand_dict, indent=4, ensure_ascii=False))
                
    with open(save_path_1, 'w', encoding='utf-8') as f:
        f.write(json.dumps(solution_dict, indent=4, ensure_ascii=False))
    with open(save_path_2, 'w', encoding='utf-8') as f:
        f.write(json.dumps(solution_expand_dict, indent=4, ensure_ascii=False))
    
def solve_iteration():
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())
    save_path = f"datasets/processed_data/fgo_{split}_solution.json"
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    solution_dict = {}
    for problem_idx in tqdm(keys):
        problem_idx = int(problem_idx)
        solver = Interactor(dl.predicate_GDL, dl.theorem_GDL, t_info)
        
        problem_CDL = dl.get_problem(pid=problem_idx)
        solver.load_problem(problem_CDL)

        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            solver.apply_theorem(t_name, t_branch, t_para)
            
        solver.problem.check_goal()
        
        conditions = solver.problem.condition
        target_condition = conditions.items[-1]
        
        condition_graph = ConditionGraph(conditions.items)
        condition_graph.construct_graph()

        info_dict = display_solution(condition_graph, target_condition, solver.problem.goal)
        solution_dict[problem_idx] = info_dict
        
        if problem_idx % 100 == 0:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(solution_dict, indent=4, ensure_ascii=False))
                
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(solution_dict, indent=4, ensure_ascii=False))
    
def draw_iteration():
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())
    keys = random.sample(keys, 100)
    save_dir = "imgs"
    
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    for problem_idx in tqdm(keys):
        problem_idx = int(problem_idx)
        solver = Interactor(dl.predicate_GDL, dl.theorem_GDL, t_info)
        
        problem_CDL = dl.get_problem(pid=problem_idx)
        solver.load_problem(problem_CDL)

        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            solver.apply_theorem(t_name, t_branch, t_para)
            
        solver.problem.check_goal()
        
        conditions = solver.problem.condition
        target_condition = conditions.items[-1]
        
        condition_graph = ConditionGraph(conditions.items)
        condition_graph.construct_graph()
        draw_graph(condition_graph, 
                   idx=problem_idx,
                   target_condition=target_condition, 
                   img_dir=save_dir)
        
def check_type():
    from collections import Counter
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())
    save_path = f"datasets/processed_data/fgo_{split}_solution.json"
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    type_lst = []
    for problem_idx in tqdm(keys):
        problem_idx = int(problem_idx)
        solver = Interactor(dl.predicate_GDL, dl.theorem_GDL, t_info)
        
        problem_CDL = dl.get_problem(pid=problem_idx)
        solver.load_problem(problem_CDL)
        
        problem = solver.problem.parsed_problem_CDL
        type = problem['parsed_cdl']['goal']['type']
        type_lst.append(type)

        # print(goal)
        if type != 'value':
            pass
        
    counted_type = Counter(type_lst)
    print(counted_type)
        
def setup_seed():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    setup_seed()
    solve_test()
    # check_type()
    # solve_iteration()
    # draw_iteration()
    # solve_and_expand_iteration()