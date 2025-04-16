# Copyright (c) 2025 Yicheng Pan
# Licensed under the MIT License. See LICENSE file in the project root for details.

import json
import os
import random
import re
import time
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
import sympy
from sympy import Eq
from tqdm import tqdm
import traceback

from formalgeo.data import DatasetLoader
from formalgeo.parse import inverse_parse_one, parse_theorem_seqs
from formalgeo.problem import Problem
from formalgeo.problem.condition import Goal
from formalgeo.solver import Interactor

from graph import ConditionGraph, ConditionNode, display_solution, draw_graph
from target_finder import TargetFinder
from utils.formulate import clause_to_nature_language
from utils.symbolic import move_subtractions_to_rhs
from utils.tools import read_json, write_json
from func_timeout import func_timeout, FunctionTimedOut

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

def create_problem_prove(target: Tuple, problem: Problem, natural_template: Dict):
    target_clause = inverse_parse_one(
        target[0], target[1], problem
    )
    target_statement = clause_to_nature_language(
        [target_clause],
        natural_template,
        upper=False,
        replace_sym=True,
        replace_sym_mode='math'
    )
    return f'Prove that {target_statement[0]}.'
    

def solve_one_test():

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


def expand_one_sample_with_timeout(
        problem_idx,
        problem_CDL,
        predicate_GDL,
        theorem_GDL,
        t_info,
        natural_template,
        save_dir):
    try:
        result = func_timeout(
            200, 
            # 300,
            expand_one_sample, 
            args=(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir)
        )
        return result
    # except Exception as e:
    except FunctionTimedOut as e:
        tb = traceback.format_exc()
        print(tb)
        print(f"Error: {str(e)}")
        return False
    
def solve_one_sample_with_timeout(
        problem_idx,
        problem_CDL,
        predicate_GDL,
        theorem_GDL,
        t_info,
        natural_template,
        save_dir):
    try:
        result = func_timeout(
            300, 
            solve_one_sample, 
            args=(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir)
        )
        return result
    # except Exception as e:
    except FunctionTimedOut as e:
        tb = traceback.format_exc()
        print(tb)
        print(f"Error: {str(e)}")
        return False

def solve_one_sample(
    problem_idx,
    problem_CDL,
    predicate_GDL,
    theorem_GDL,
    t_info,
    natural_template,
    save_dir
):
    # if os.path.exists(f"{save_dir}/{problem_idx}.json"):
    #     return True
    search_start = time.time()
    solver = Interactor(predicate_GDL, theorem_GDL, t_info)
    
    solver.load_problem(problem_CDL)
    for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
        solver.apply_theorem(t_name, t_branch, t_para)
    
    solver.problem.check_goal()
    if not solver.problem.goal.solved:
        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            solver.apply_theorem(t_name, t_branch, t_para)
        solver.problem.check_goal()

    search_time = time.time() - search_start
    
    # exit when not solved
    if not solver.problem.goal.solved:
        print(f'Problem {problem_idx} not solved.')
        return 
    
    if solver.problem.goal.type == 'algebra':
        target_premise = solver.problem.goal.premise
        if len(target_premise) > 1:
            target_expr = solver.problem.goal.item - solver.problem.goal.answer
            solver.problem.add('Equation', target_expr, target_premise, ('extended', None, None))
            target_id = solver.problem.condition.get_id_by_predicate_and_item('Equation', target_expr)
        elif len(target_premise) == 1:
            target_id = target_premise[0]
        else:
            raise ValueError(len(target_premise))

        ori_target = solver.problem.condition.items[target_id]

    elif solver.problem.goal.type == 'logic':
        print('-------problem_idx-------: ', problem_idx)
        target_predicate = solver.problem.goal.item
        target_id = solver.problem.condition.get_id_by_predicate_and_item('Logic', target_predicate)
        ori_target = solver.problem.condition.items[target_id]
    
    # construct graph
    condition_graph = ConditionGraph(solver.problem.condition.items)
    condition_graph.construct_graph()
    
    # solution to original target    
    (
        solution_str, 
        solution_formal_dict, 
        theorems_formal, 
        sub_nodes, 
        too_complex
    ) = TargetFinder.find_solution_for_target(
        solver.problem, 
        condition_graph, 
        ori_target, 
        natural_template,
        solver.parsed_theorem_GDL,
        expand_flag=True
    )

    if not solver.problem.goal.solved:
        print(f'{problem_idx} not solved')
        
    data_info = {
        "key": problem_idx,
        "solved": solver.problem.goal.solved,
        "source": problem_CDL['source'],
        "construction_cdl": problem_CDL['construction_cdl'],
        "text_cdl": problem_CDL['text_cdl'],
        "image_cdl": problem_CDL['image_cdl'],
        "positions": [],
        "goal_cdl": problem_CDL['goal_cdl'],
        "search_time": search_time,
        "theorems": problem_CDL['theorem_seqs'],
        "llm_info": {
            "key": problem_idx,
            "problem_level": len(problem_CDL['theorem_seqs']),
            "problem_text": problem_CDL['problem_text_en'],
            "problem_answer": problem_CDL['problem_answer'],
            "solution_dict": solution_formal_dict,
            "solution_str": solution_str,
            "caption_str": ""
        }
    }

    with open(f"{save_dir}/{problem_idx}.json", 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=4, ensure_ascii=False)

    return 

def expand_one_sample(
    problem_idx,
    problem_CDL,
    predicate_GDL,
    theorem_GDL,
    t_info,
    natural_template,
    save_dir,
    debug=False
):
    if os.path.exists(f"{save_dir}/{problem_idx}_0.json"):
        return True
    solver = Interactor(predicate_GDL, theorem_GDL, t_info)
    
    solver.load_problem(problem_CDL)
    # solver.expand_conditions()

    for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
        solver.apply_theorem(t_name, t_branch, t_para)
    
    solver.problem.check_goal()
    if not solver.problem.goal.solved:
        for t_name, t_branch, t_para in parse_theorem_seqs(problem_CDL["theorem_seqs"]):
            solver.apply_theorem(t_name, t_branch, t_para)
        solver.problem.check_goal()

    # expand
    solver.expand_conditions()
    
    if solver.problem.goal.type == 'algebra':
        target_premise = solver.problem.goal.premise
        if len(target_premise) > 1:
            target_expr = solver.problem.goal.item - solver.problem.goal.answer
            solver.problem.add('Equation', target_expr, target_premise, ('extended', None, None))
            target_id = solver.problem.condition.get_id_by_predicate_and_item('Equation', target_expr)
        elif len(target_premise) == 1:
            target_id = target_premise[0]
        else:
            raise ValueError(len(target_premise))

        ori_target = solver.problem.condition.items[target_id]

    elif solver.problem.goal.type == 'logic':
        print('-------problem_idx-------: ', problem_idx)
        target_predicate = solver.problem.goal.item
        target_id = solver.problem.condition.get_id_by_predicate_and_item('Logic', target_predicate)
        ori_target = solver.problem.condition.items[target_id]


    allocater_states = {
        "p_pos": None,
        "lines": None,
        "circles": None,
        "points_on_circle": None,
        "clauses": None
    }
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}

    target_finder = TargetFinder(predicate_GDL, theorem_GDL, 
                                t_info, t_freq_info, allocater_states, 
                                problem_CDL['text_cdl'] ,problem_CDL['construction_cdl'],
                                problem_CDL['image_cdl'],
                                problem_id=0, replace_characters=False,
                                solver_type='formalgeo', predicate_num=2, 
                                debug=debug)

    target_finder.solver = solver
    condition_graph = ConditionGraph(target_finder.solver.problem.condition.items)
    condition_graph.construct_graph()

    conditions_to_sample = solver.problem.condition.items[-15:]
    conditions_to_sample = [t for t in conditions_to_sample if t != ori_target]
    # filter 1
    new_targets = TargetFinder.targets_filter_1(
        conditions_to_sample,
        target_finder.solver.problem.condition.value_of_sym
    )
    # filter 2
    new_targets = TargetFinder.targets_filter_2(
        new_targets,
        strict=True
    )
    if len(new_targets) == 0:
        return 

    theorems_for_targets = {}
    solution_for_targets = {}
    solution_dict_for_targets = {}
    level_for_targets = {}
    for target in new_targets:
        (
            solution_str, 
            solution_formal_dict, 
            theorems_formal, 
            sub_nodes, 
            too_complex
        ) = TargetFinder.find_solution_for_target(
            target_finder.solver.problem,
            condition_graph, 
            target, 
            natural_template,
            target_finder.solver.parsed_theorem_GDL,
            expand_flag=True
        )
        if too_complex:
            continue
        theorems_for_targets[target] = [v['theorem'] for v in solution_formal_dict.values() if v['theorem'] != None]
        solution_for_targets[target] = solution_str
        level_for_targets[target] = len(theorems_for_targets[target])
        solution_dict_for_targets[target] = solution_formal_dict

    # filter 3
    new_targets, idx_for_targets = target_finder.targets_filter_3(
        new_targets, theorems_for_targets, target_finder.max_depth
    )
    targets_dict = TargetFinder.targets_into_groups(new_targets)

    chosen_targets = []
    type_list = ['line', 'angle', 'prove']
    # type_list = ['line', 'angle'， ‘]
    while any([len(targets_dict[t]) > 0 for t in type_list]):  # 只要有一个列表不为空就继续循环
        for type_i in type_list:
            if targets_dict[type_i]:  # 检查列表是否为空
                element = targets_dict[type_i].pop(0)
                chosen_targets.append(element)
            else:
                pass
    
    source_key = problem_CDL['source']
    if 'GeoQA' in source_key: # less targets for geoqa
        chosen_targets = chosen_targets[:3]
    else:
        chosen_targets = chosen_targets[:6]

    used_symbols = [str(s) for s in list(target_finder.solver.problem.condition.value_of_sym.keys())]

    # create question for every target
    for i, chosen_target in enumerate(chosen_targets):
        chosen_thoerems = theorems_for_targets[chosen_target]
        chosen_solution = solution_for_targets[chosen_target]
        chosen_solution_dict = solution_dict_for_targets[chosen_target]
        (
            conclusion, 
            add_cdls,
            add_conditions, 
            target_value, 
            target_cdl, 
            problem_text, 
            cap_str
        ) = target_finder.create_question(chosen_target, 'image_based', used_symbols=used_symbols)

        if len(add_conditions) != 0:
            chosen_solution += f"\n<because> {', '.join(add_conditions)}, <therefore> {conclusion}."
        else:
            chosen_solution += f"\n<therefore> {conclusion}."

        # problem_text = problem_CDL['problem_text_en'].split('Find')[0] + problem_text
        data_info = {
            "key": problem_idx,
            "solved": True,
            "source": problem_CDL['source'],
            "construction_cdl": problem_CDL['construction_cdl'],
            "text_cdl": problem_CDL['text_cdl'],
            "image_cdl": problem_CDL['image_cdl'],
            "positions": [],
            "goal_cdl": '',
            "theorems": chosen_thoerems,
            "llm_info": {
                "key": problem_idx,
                "problem_level": level_for_targets[chosen_target],
                "problem_text": problem_text,
                "problem_answer": str(target_value),
                "solution_dict": chosen_solution_dict,
                "solution_str": chosen_solution,
                "caption_str": ""
            }
        }
        with open(f"{save_dir}/{problem_idx}_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
            
    return
                
def solve_main(split="train"):
    # save_dir = f"datasets/fgo_search_{split}"
    save_dir = f"datasets/fgo_search_{split}_v5_ori"
    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())

    # exist_keys = 3242
    # keys = keys[3900:]
    # regen_keys = read_json('D:/Desktop/GeoExpand/jsons/regen_keys_v2.json')
    # keys = [k for k in keys if k in regen_keys]

    num_process = 10
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
    
    pool = Pool(num_process)
    results = []

    
    with tqdm(total=len(keys), desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for i, problem_idx in enumerate(keys):
            
            # problem_CDL = data[problem_idx]
            problem_idx = int(problem_idx)
            problem_CDL = dl.get_problem(problem_idx)
            res = pool.apply_async(
                # expand_one_sample,
                solve_one_sample_with_timeout,
                args=(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir),
                callback=update
            )
            results.append(res)
        for r in results:
            r.wait()
        # if os.path.exists(f"{save_dir}/{problem_idx}.json"):
        #     tmp_data = json.load(open(f"{save_dir}/{problem_idx}.json", 'r', encoding='utf-8'))
        #     if 'solved' in tmp_data:
        #         continue

    print("End for solve.")

def expand_main(split="train"):
    # save_dir = f"datasets/fgo_search_{split}"
    save_dir = f"datasets/fgo_search_{split}_v5_exp"
    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())

    # exist_keys = 3242
    # keys = keys[3900:]
    # regen_keys = read_json('D:/Desktop/GeoExpand/jsons/regen_keys_v2.json')
    # keys = [k for k in keys if k in regen_keys]

    num_process = 6
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
    
    pool = Pool(num_process)
    results = []

    
    with tqdm(total=len(keys), desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for i, problem_idx in enumerate(keys):
            
            # problem_CDL = data[problem_idx]
            problem_idx = int(problem_idx)
            problem_CDL = dl.get_problem(problem_idx)
            res = pool.apply_async(
                # expand_one_sample,
                expand_one_sample_with_timeout,
                args=(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir),
                callback=update
            )
            results.append(res)
        for r in results:
            r.wait()
        # if os.path.exists(f"{save_dir}/{problem_idx}.json"):
        #     tmp_data = json.load(open(f"{save_dir}/{problem_idx}.json", 'r', encoding='utf-8'))
        #     if 'solved' in tmp_data:
        #         continue

    print("End for expand.")
        
def solve_test():
    data_path = f"datasets/processed_data/fgo_train.json"
    save_dir = f"datasets/fgo_train_search_debug"
    predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))
    os.makedirs(save_dir, exist_ok=True)
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())

    # regen_keys = read_json('D:/Desktop/GeoExpand/jsons/regen_keys_v2.json')
    # keys = [k for k in keys if k in regen_keys]

    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
 
    results = []
    for problem_idx in tqdm(keys):
        # if int(problem_idx) < 19:
        #     continue
        
        problem_CDL = data[problem_idx]
        problem_idx = int(problem_idx)
        res = solve_one_sample(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir)
        results.append(res)

def expand_test():
    data_path = f"datasets/processed_data/fgo_train.json"
    save_dir = f"datasets/fgo_train_expand_debug"
    os.makedirs(save_dir, exist_ok=True)
    # predicate_GDL = json.load(open("json/predicate_GDL_for_search.json", 'r', encoding='utf-8'))
    # theorem_GDL = json.load(open("json/theorem_GDL_for_search.json", 'r', encoding='utf-8'))
    predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))
    os.makedirs(save_dir, exist_ok=True)
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())

    # regen_keys = read_json('D:/Desktop/GeoExpand/jsons/regen_keys_v2.json')
    # keys = [k for k in keys if k in regen_keys]

    # t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
 
    results = []
    for problem_idx in tqdm(keys):
        # if int(problem_idx) < 19:
        #     continue
        
        problem_CDL = data[problem_idx]
        problem_idx = int(problem_idx)
        res = expand_one_sample(problem_idx, problem_CDL, predicate_GDL, theorem_GDL, t_info, natural_template, save_dir, debug=True)
        results.append(res)
 

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

        
def setup_seed():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

def check_goal():
    data_path = f"datasets/processed_data/fgo_train.json"
    save_dir = f"datasets/fgo_train_search_debug"
    predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))
    os.makedirs(save_dir, exist_ok=True)
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    keys = list(data.keys())

    # regen_keys = read_json('D:/Desktop/GeoExpand/jsons/regen_keys_v2.json')
    # keys = [k for k in keys if k in regen_keys]

    t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
 
    results = []
    for problem_idx in tqdm(keys):
        # if int(problem_idx) < 19:
        #     continue
        
        problem_CDL = data[problem_idx]
        problem_idx = int(problem_idx)
        if problem_CDL['goal_cdl'] == '':
           print(problem_idx)


if __name__ == '__main__':
    setup_seed()
    # solve_test()
    # check_type()
    # solve_iteration()
    # draw_iteration()
    solve_main(split='val')
    # expand_test()
    # check_goal()
    expand_main()