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
from formalgeo_v2.parse.basic import parse_equal_predicate
from formalgeo.problem import Problem
from formalgeo.problem.condition import Goal
from formalgeo.solver import Interactor

from graph import ConditionGraph, ConditionNode, display_solution, draw_graph
from target_finder import TargetFinder
from utils.formulate import clause_to_nature_language
from utils.symbolic import move_subtractions_to_rhs, parse_clause
from utils.tools import read_json, write_json
from func_timeout import func_timeout, FunctionTimedOut



def search_one_sample_with_timeout(
        image_idx,
        info,
        predicate_GDL,
        theorem_GDL,
        t_info,
        t_freq_info,
        save_dir,
        natural_template,
        debug=False
    ):
    try:
        result = func_timeout(
            300,
            search_one_sample, 
            args=(
                image_idx,
                info,
                predicate_GDL,
                theorem_GDL,
                t_info,
                t_freq_info,
                save_dir,
                natural_template,
                debug
            )
        )
        return result
    # except Exception as e:
    except FunctionTimedOut as e:
        tb = traceback.format_exc()
        print(tb)
        print(f"Error: {str(e)}")
        return False
    

def search_one_sample(
    image_idx,
    info,
    predicate_GDL,
    theorem_GDL,
    t_info,
    t_freq_info,
    save_dir,
    natural_template,
    debug=False
):
    image_key = image_idx.split('.png')[0]
    geo_states = {
        "p_pos": info['p_pos'],
        "lines": info['lines'],
        "circles": info['circles']
    }
        
    target_finder = TargetFinder(
        predicate_GDL,
        theorem_GDL,
        t_info,
        t_freq_info,
        geo_states,
        info['text_cdls'],
        info['construct_cdls'],
        info['image_cdls'],
        replace_characters=False,
        solver_type='formalgeo',
        debug=debug
    )

    if os.path.exists(f"{save_dir}/{image_key}_0.json"):
        return True

    
    target_finder.solver.init_search(target_finder.problem_CDL)
    target_finder.solver.search()

    # construct condition graph
    condition_graph = ConditionGraph(target_finder.solver.problem.condition.items)
    condition_graph.construct_graph()

    (
        targets_dict,
        theorems_for_targets,
        solution_for_targets,
        solution_dict_for_targets
    ) = target_finder.filter_conditions(condition_graph, strict=True)

    chosen_targets = []
    type_list = ['value', 'line', 'angle']
    # type_list = ['line', 'angle'， ‘]
    while any([len(targets_dict[t]) > 0 for t in type_list]):  # 只要有一个列表不为空就继续循环
        for type_i in type_list:
            if targets_dict[type_i]:  # 检查列表是否为空
                element = targets_dict[type_i].pop(0)
                chosen_targets.append(element)
            else:
                pass

    chosen_targets = chosen_targets[:5] + targets_dict['prove'][:2]
    used_symbols = [str(s) for s in list(target_finder.solver.problem.condition.value_of_sym.keys())]
    a = 1
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
        ) = target_finder.create_question(chosen_target, 
                                          used_symbols=used_symbols,
                                          angle_ids=info['angle_ids'])
        if conclusion is None:
            continue
        if len(add_conditions) != 0:
            chosen_solution += f"\n<because> {', '.join(add_conditions)}, <therefore> {conclusion}."
        else:
            chosen_solution += f"\n<therefore> {conclusion}."

        
        # problem_text = problem_CDL['problem_text_en'].split('Find')[0] + problem_text
        data_info = {
            "key": image_key,
            "solved": True,
            "source": f"pgps9k_{image_key}",
            "construction_cdl": info['construct_cdls'],
            "text_cdl": info['text_cdls'],
            "image_cdl": info['image_cdls'],
            "positions": [],
            "goal_cdl": '',
            "theorems": chosen_thoerems,
            "llm_info": {
                "key": image_key,
                "problem_level": len(chosen_thoerems),
                "problem_text": problem_text,
                "problem_answer": str(target_value),
                "solution_dict": chosen_solution_dict,
                "solution_str": chosen_solution,
                "cap_str": cap_str
            }
        }
        with open(f"{save_dir}/{image_key}_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
            
    return

    

def search_main(split="train"):
    data_path = f"datasets/pgps/train_formal_img.json"
    save_dir = f"datasets/pgps/train_search"
    keys_ignore_path = f"datasets/pgps/lack_pos_keys.json"
    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("json/predicate_GDL_for_search.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("json/theorem_GDL_for_search.json", 'r', encoding='utf-8'))
    # predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    # theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))

    data = read_json(data_path)
    ignore_keys = read_json(keys_ignore_path)
    keys = list(data.keys())
    keys = [k for k in keys if k not in ignore_keys]

    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))

    cnt = 0

    num_process = 6
    pool = Pool(num_process)
    results = []

    
    with tqdm(total=len(keys), desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for i, image_idx in enumerate(keys):
            info = data[image_idx]
            res = pool.apply_async(
                search_one_sample_with_timeout,
                args=(
                    image_idx, 
                    info, 
                    predicate_GDL, 
                    theorem_GDL, 
                    t_info, 
                    t_freq_info, 
                    save_dir, 
                    natural_template,
                ),
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
        

def search_test():
    data_path = f"datasets/pgps/train_formal_img.json"
    save_dir = f"datasets/pgps/train_search"
    keys_ignore_path = f"datasets/pgps/lack_pos_keys.json"
    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("json/predicate_GDL_for_search.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("json/theorem_GDL_for_search.json", 'r', encoding='utf-8'))
    # predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    # theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))

    data = read_json(data_path)
    ignore_keys = read_json(keys_ignore_path)
    keys = list(data.keys())
    keys = [k for k in keys if k not in ignore_keys]

    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
 
    results = []
    cnt = 0
    image_dir_1 = 'D:/Desktop/资源/几何答题/UniAll/imgs/pgps9k'
    image_dir_2 = 'D:/Desktop/资源/几何答题/UniAll/imgs/pgps9k_annoted'

    # keys = random.sample(keys, 10)
    fail_cases = {}
    for image_idx in tqdm(keys):
        cnt += 1
        # if cnt < 55:
        #     continue
        info = data[image_idx]
        print(f'------------- {cnt} -------------')
        if os.path.exists(f'{image_dir_2}/{image_idx}'):
            print(f'{image_dir_2}/{image_idx}')
        else:
            print(f'{image_dir_1}/{image_idx}')
        try:
            res = search_one_sample(
                image_idx, 
                info, 
                predicate_GDL, 
                theorem_GDL, 
                t_info, 
                t_freq_info, 
                save_dir, 
                natural_template,
                debug=True
            )
            results.append(res)
        except Exception as e:
            tb = traceback.format_exc() 
            fail_cases[image_idx] = tb
 

        
def setup_seed():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

def get_content(clause):
    if 'Equal(' in clause:
        eq_tree, attrs = parse_equal_predicate(clause)
        left_tree, right_tree = eq_tree[1][0], eq_tree[1][1]
        left_ps = get_content_in_tree(left_tree)
        right_ps = get_content_in_tree(right_tree)
        points = list(set(left_ps + right_ps))
        return points
    else:
        _, items = parse_clause(clause)
        points = set(list(''.join(items)))
        return list(points)
    
def get_content_in_tree(tree):
    if isinstance(tree, str):
        if tree.isalpha() and tree.isupper():
            a = 1
        return []
    if len(tree) == 1:
        return tree[0]
    name = tree[0]
    if name in ['Add', 'Mul', 'Sub', 'Div', 'Pow', 'Mod']:
        left_ps = get_content_in_tree(tree[1][0])
        right_ps = get_content_in_tree(tree[1][1])
        points = list(set(left_ps + right_ps))
        return points
    elif len(tree) == 2:
        if tree[0] == 'Sqrt':
            return []
        return list(tree[1])

def stats():
    # data_path = f"datasets/pgps/train_formal_img.json"
    data_path = 'D:/Desktop/资源/几何答题/UniAll/jsons/pgps9k/train_formal_img.json'
    data = json.load(open(data_path, 'r', encoding='utf-8'))

    img_keys_lack_pos = set()
    for k, v in tqdm(data.items()):
        ps_has_pos = list(v['p_pos'].keys())
        clauses = v['image_cdls'] + v['construct_cdls'] + v['text_cdls']
        p_list = []
        for clause in clauses:
            p_list += get_content(clause)
        ps_in_clause = list(set(p_list))

        if len(ps_has_pos) < len(ps_in_clause):
            img_keys_lack_pos.add(k)

    print('Lack pos keys: ', len(img_keys_lack_pos))
    write_json('datasets/pgps/lack_pos_keys.json', sorted(img_keys_lack_pos))


if __name__ == '__main__':
    setup_seed()
    # solve_test()
    # check_type()
    # solve_iteration()
    # draw_iteration()
    # expand_test()
    # check_goal()
    # search_test()
    search_main()
    # stats()