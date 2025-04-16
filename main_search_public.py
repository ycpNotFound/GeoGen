import json
import os
import random
import re
import time
from multiprocessing import Pool

import argparse
import numpy as np
import sympy
from sympy import Eq
from tqdm import tqdm
import traceback


from graph import ConditionGraph, ConditionNode, display_solution, draw_graph
from target_finder import TargetFinder
from utils.formulate import clause_to_nature_language
from utils.symbolic import move_subtractions_to_rhs, parse_clause
from utils.tools import read_json, write_json, setup_seed
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
        # print(tb)
        # print(f"Error: {str(e)}")
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
        "p_pos": info['p_pos'] if 'p_pos' in info else None,
        "lines": info['lines'] if 'lines' in info else None,
        "circles": info['circles'] if 'circles' in info else None
    }
        
    target_finder = TargetFinder(
        predicate_GDL,
        theorem_GDL,
        t_info,
        t_freq_info,
        geo_states,
        info['text_cdl'],
        info['construction_cdl'],
        info['image_cdl'],
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
    
    if sum([len(v) for v in targets_dict.values()]) == 0:
        (
            targets_dict,
            theorems_for_targets,
            solution_for_targets,
            solution_dict_for_targets
        ) = target_finder.filter_conditions(condition_graph, strict=False)

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
        ) = target_finder.create_question(
            chosen_target, 
            used_symbols=used_symbols,
            angle_ids=info['angle_ids'] if 'angle_ids' in info else {}
        )
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
            "construction_cdl": info['construction_cdl'],
            "text_cdl": info['text_cdl'],
            "image_cdl": info['image_cdl'],
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

    

def search_main(args):
    data = get_data(args.dataset_name)

    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("json/predicate_GDL_for_search.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("json/theorem_GDL_for_search.json", 'r', encoding='utf-8'))
    # predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    # theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))

    keys = list(data.keys())
    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))

    num_process = args.num_process
    pool = Pool(num_process)
    results = []

    
    with tqdm(total=len(keys), desc="Searching") as pbar:
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

    print("End for expand.")


def search_test(args):
    data = get_data(args.dataset_name)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    predicate_GDL = json.load(open("json/predicate_GDL_for_search.json", 'r', encoding='utf-8'))
    theorem_GDL = json.load(open("json/theorem_GDL_for_search.json", 'r', encoding='utf-8'))
    # predicate_GDL = json.load(open("datasets/formalgeo7k/gdl/predicate_GDL.json", 'r', encoding='utf-8'))
    # theorem_GDL = json.load(open("datasets/formalgeo7k/gdl/theorem_GDL.json", 'r', encoding='utf-8'))

    keys = list(data.keys())


    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    natural_template = json.load(open("json/predicates_to_nature_language.json", 'r', encoding='utf-8'))
 
    results = []
    cnt = 0


    fail_cases = {}
    for image_idx in tqdm(keys):
        cnt += 1
        # if cnt < 55:
        #     continue
        info = data[image_idx]
        print(f'------------- {cnt} -------------')

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
        # except Exception as e:
        #     tb = traceback.format_exc() 
        #     fail_cases[image_idx] = tb
 
def get_data(dataset_name):
    assert dataset_name in ['geo3k', 'pgps9k']
    if dataset_name == 'geo3k':
        data = read_json('datasets_info/img_info_geo3k_train.json')
    elif dataset_name == 'pgps9k':
        data = read_json(f"datasets_info/img_info_pgps9k_train.json")
    else:
        raise KeyError(dataset_name)
    return data


def get_args():
    parser = argparse.ArgumentParser(description="Search for Geometry3K / PGPS9K")

    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='pgps9k',
                        # default='geo3k', 
                        choices=['pgps9k', 'geo3k'])
    parser.add_argument('--save_dir', type=str, default='results_expand/geo3k')
    parser.add_argument('--num_process', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    setup_seed(args.seed)
    if args.debug:
        search_test(args)
    else:
        search_main(args)

if __name__ == '__main__':

    main()