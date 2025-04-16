
import itertools
import json
import os
import random
import traceback
from multiprocessing import Pool

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from allocator import Allocator
from generator import ClauseGenerator
from plotter import Plotter
from utils.preset import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,
                          PRESET_COLOR_PROBS, PRESET_COLORS)
from utils.tools import setup_seed


def generate_one_sample(predicate_GDL, 
                        theorem_GDL,
                        predicate_base, 
                        predicate_rel, 
                        n_more_lines, 
                        color_config, 
                        fig_dir, 
                        fig_idx):
    try:
        cg = ClauseGenerator(predicate_GDL, theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            predicate_base, 
            predicate_rel, 
            n_more_lines=n_more_lines
        )
        allocator = Allocator(cg.states, c_cdls, t_cdls, allocate_value=True)
        allocator.allocate() 
        plotter = Plotter(
            allocator.states,
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls'],
            allocator.formulated_cdls['image_cdls'],
            replace_characters=True,
            color_config=color_config,
            min_side=300,
            max_side=350
        )
        plotter.plot()
        fig_name = f"{fig_idx}.png"
        resize_ratio = np.random.choice(
            [0.4, 0.6, 0.8, 1], p=[0.2, 0.3, 0.3, 0.2]
        )
        plotter.save_fig(
            fig_dir=fig_dir, 
            fig_name=fig_name, 
            resize_ratio=resize_ratio
        )
        task_info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "n_more_lines": n_more_lines,
            "color_config": color_config,
        }
        data_info = {
            "key": fig_idx,
            "construction_cdl": plotter.construct_cdls,
            "text_cdl": plotter.text_cdls, 
            "image_cdl": plotter.image_cdls,
            "caption_str": plotter.caption_str,
            "positions": plotter.p_pos
        }
        return (True, data_info)
    except Exception as e:
        print(f"===== Error Occured: {fig_idx} =====")
        tb = traceback.format_exc()
        print(tb)
        task_info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "n_more_lines": n_more_lines,
            "color_config": color_config,
            "error_message": tb
        }
        return (False, task_info)
    
def generate_one_sample_with_timeout(
                          predicate_GDL, 
                          theorem_GDL, 
                          predicate_base, 
                          predicate_rel, 
                          n_more_lines,
                          color_config,
                          fig_dir, 
                          fig_idx):
    try:
        result = func_timeout(
            200, 
            generate_one_sample, 
            args=(predicate_GDL, theorem_GDL, predicate_base, predicate_rel, 
                  n_more_lines, color_config, fig_dir, fig_idx)
        )
        return result
    
    except FunctionTimedOut:
        print(f"===== Time Out: {fig_idx} =====")
        info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "n_more_lines": n_more_lines,
            "color_config": color_config,
            "error_message": "timeout"
        }
        return (False, info)


def run_task(seed, 
             task_name, 
             input_args_list,
             num_process):
    # seed, task_name, predicate_base_combs, predicate_rel_combs = task_1()
    setup_seed(seed)
    
    json_file_path = f"geo_gen/{task_name}/annotations.json"
    failure_cases_path = f"geo_gen/{task_name}/failure_cases.json"
    fig_dir = f"geo_gen/{task_name}/imgs"
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Start Generation ...")
    
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    cnt = 0
    total_iterations = len(input_args_list)
    result_info = []
    
    # multiprocess generate
    pool = Pool(num_process)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for args in input_args_list:
            pred_base, pred_rel, n_more_lines, color_config = args
            result = pool.apply_async(
                generate_one_sample_with_timeout, 
                args=(predicate_GDL, theorem_GDL, pred_base, pred_rel, 
                      n_more_lines, color_config, fig_dir, cnt),
                callback=update)
            result_info.append(result)
            cnt += 1

        for r in result_info:
            r.wait()

    # save success and failure cases
    result_info = [r.get() for r in result_info]
    failure_cases = []
    info_dict = {}
    for success, info in result_info:
        if success:
            key = info.pop("key")
            info_dict[key] = info
        else:
            failure_cases.append(info)
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(info_dict, indent=4, ensure_ascii=False))
                
    print(f"Success Count: {cnt - len(failure_cases)} / {cnt}")
    print(f"Failure Count: {len(failure_cases)} / {cnt}")
    print("End for Generation. Re-generate for Failure Cases ...")
    
    # re-generate for failure cases
    failure_count = {}
    failure_dict = {}
    
    with tqdm(total=len(failure_cases), desc="Processing") as pbar:
        while True:
            if len(failure_cases) == 0:
                break
            init_info = failure_cases.pop(0)
            predicate_base = init_info['pred_base']
            predicate_rel = init_info['pred_rel']
            n_more_lines = init_info['n_more_lines']
            color_config = init_info['color_config']
            key = init_info["key"]
            
            if key not in failure_count:
                failure_count[key] = 0
                
            result = generate_one_sample_with_timeout(
                predicate_GDL, theorem_GDL, predicate_base, predicate_rel, 
                n_more_lines, color_config,fig_dir, key)
            
            success, info = result
            if success:
                info_dict[key] = info
                pbar.update()
            else: # end if fail accumulated for 3 times
                if failure_count[key] >= 2:
                    failure_dict[key] = init_info
                    print('Fail for 3 times already: ')
                    print(f'Predicates Base:')
                    for pred in predicate_base:
                        print(f"\t\t{pred}")
                    print(f'Predicates Rel:')
                    for pred in predicate_rel:
                        print(f"\t\t{pred}")
                    pbar.update()
                else:
                    failure_cases.append(info)
                    failure_count[key] += 1

    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(info_dict, indent=4, ensure_ascii=False))
    with open(failure_cases_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(failure_dict, indent=4, ensure_ascii=False))
    
    print("End for Generation.")


def build_input_args(pred_base_combs, 
                     pred_rel_combs, 
                     n_more_lines,
                     sampling_num):
    input_args = []
    for predicate_base in pred_base_combs:
        for predicate_rel in pred_rel_combs:
            for _ in range(sampling_num):
                color_config = np.random.choice(
                    PRESET_COLORS, p=PRESET_COLOR_PROBS)
                input_args.append(
                    (predicate_base, predicate_rel, n_more_lines, color_config)
                )
    return input_args

def task_0():
    seed = 1234
    task_name = "geo_gen_ENT_1_REL_0"
    input_args_list = []
    n_more_lines= 0     # add n more lines for figure
    sampling_num = 3    # repeat for one set of preds, may vary from colors
    num_process = 1     # num of process to generare
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT + PREDICATES_REL_2, 1))
    pred_rel_combs = [[]]
    input_args_list += build_input_args(pred_base_combs, 
                                        pred_rel_combs, 
                                        n_more_lines=n_more_lines,
                                        sampling_num=sampling_num)

    
    print(f'======== Task: {task_name} ========')
    return seed, task_name, input_args_list, sampling_num, num_process

def task_1():
    seed = 1234
    task_name = "geo_gen_ENT_1_REL_2"
    input_args_list = []
    num_process = 12
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=1)
    print('Num: ', len(input_args_1))

    
    input_args_list = input_args_1
    print(f'======== Task: {task_name}, Num: {len(input_args_list)} ========')
    return seed, task_name, input_args_list, num_process

def task_2():
    seed = 1234
    task_name = "geo_gen_ENT_1_REL_3"
    input_args_list = []
    num_process = 12
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 3))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=1)
    print('Num: ', len(input_args_1))

    
    input_args_list = input_args_1
    print(f'======== Task: {task_name}, Num: {len(input_args_list)} ========')
    return seed, task_name, input_args_list, num_process

def task_stage_1():
    # seed = 114
    
    input_args_list = []

    num_process = 10
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT + PREDICATES_REL_2, 1))
    pred_rel_combs = [[]]
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=400)
    task_name_1 = "geo_gen_ENT_1_REL_0_L_0"
    seed_1 = 114
    print(f'Task: {task_name_1}', len(input_args_1))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = [[]]
    input_args_2 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=400)
    task_name_2 = "geo_gen_ENT_1_REL_0_L_1"
    seed_2 = 115
    print(f'Task: {task_name_2}', len(input_args_2))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_3 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=300)
    task_name_3 = "geo_gen_ENT_1_REL_1_L_0"
    seed_3 = 116
    print(f'Task: {task_name_3}', len(input_args_3))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_4 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=300)
    task_name_4 = "geo_gen_ENT_1_REL_1_L_1"
    seed_4 = 117
    print(f'Task: {task_name_4}', len(input_args_4))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_5 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=25)
    task_name_5 = "geo_gen_ENT_1_REL_2_L_0_part_1"
    seed_5 = 118
    print(f'Task: {task_name_5}', len(input_args_5))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_6 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    sampling_num=25)
    task_name_6 = "geo_gen_ENT_1_REL_2_L_0_part_2"
    seed_6 = 119
    print(f'Task: {task_name_6}', len(input_args_6))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_7 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=25)
    task_name_7 = "geo_gen_ENT_1_REL_2_L_1_part_1"
    seed_7 = 120
    print(f'Task: {task_name_7}', len(input_args_7))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_8 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=25)
    task_name_8 = "geo_gen_ENT_1_REL_2_L_1_part_2"
    seed_8 = 121
    print(f'Task: {task_name_8}', len(input_args_8))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_9 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=2,
                                    sampling_num=25)
    task_name_9 = "geo_gen_ENT_1_REL_2_L_2_part_1"
    seed_9 = 122
    print(f'Task: {task_name_9}', len(input_args_9))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_10 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    sampling_num=25)
    task_name_10 = "geo_gen_ENT_1_REL_2_L_2_part_2"
    seed_10 = 123
    print(f'Task: {task_name_10}', len(input_args_10))
    
    input_args_list = [
        # input_args_1, input_args_2, 
        # input_args_3, input_args_4,
        # input_args_5, 
        input_args_6,
        input_args_7, input_args_8,
        input_args_9, input_args_10,
    ]
    task_name_list = [
        # task_name_1, task_name_2, 
        # task_name_3, task_name_4,
        # task_name_5, 
        task_name_6,
        task_name_7, task_name_8,
        task_name_9, task_name_10
    ]
    seed_list = [
        # seed_1, seed_2,
        # seed_3, seed_4,
        # seed_5, 
        seed_6,
        seed_7, seed_8,
        seed_9, seed_10,
    ]
    print('Total Num: ', sum([len(args) for args in input_args_list]))
    for input_args, task_name, seed in zip(input_args_list, task_name_list, seed_list):
        print(f'======== Task: {task_name}, Num: {len(input_args)} ========')
        run_task(seed, task_name, input_args, num_process)
    # return seed, task_name, input_args_list, num_process
    
def main():
    # run_task(*task_2())
    # run_task(*task_pretrain())
    task_stage_1()


if __name__ == '__main__':
    main()
    