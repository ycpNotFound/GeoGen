
import itertools
import json
import os
import traceback
from multiprocessing import Pool

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from allocator import Allocator
from generator import ClauseGenerator
from plotter import Plotter
from target_finder import TargetFinder
from utils.preset import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,
                          PRESET_COLOR_PROBS, PRESET_COLORS)
from utils.tools import identify_image, setup_seed, write_json


def generate_one_sample(predicate_GDL, 
                        theorem_GDL,
                        predicate_base, 
                        predicate_rel, 
                        n_more_lines, 
                        color_config, 
                        fig_dir, 
                        fig_idx,
                        info_dir,
                        search_cfg):
    try:
        for i in range(5): # try 5 times to plot
            cg = ClauseGenerator(predicate_GDL, theorem_GDL)
            cg.empty_states()
            c_cdls, t_cdls = cg.generate_clauses_from_predicates(
                predicate_base, 
                predicate_rel, 
                n_more_lines=n_more_lines
            )
            allocator = Allocator(
                cg.states, c_cdls, t_cdls, 
                allocate_value=False,
                replace_chars=True
            )
            allocator.allocate()
            
            plotter = Plotter(
                allocator.states,
                allocator.formulated_cdls['text_cdls'],
                allocator.formulated_cdls['construct_cdls'],
                allocator.formulated_cdls['image_cdls'],
                replace_characters=False,
                color_config=color_config,
                stage_2=True
            )
            plotter.plot()
            fig_name = f"{fig_idx}.png"
            plotter.save_fig(fig_dir=fig_dir, fig_name=fig_name)

            plot_success, _ = identify_image(
                plotter.p_pos, plotter.fig_size)
            if plot_success:
                break
        
        goal_finder = TargetFinder(
            search_cfg['predicate_GDL'],
            search_cfg['theorem_GDL'],
            search_cfg['t_info'],
            search_cfg['t_freq_info'],
            allocator.states, 
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls'],
            allocator.image_cdls,
            replace_characters=False,
            solver_type='formalgeo',
            debug=False
        )
        info_dict_symbolic, info_dict_llm = goal_finder.formulate()
        
        task_info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "n_more_lines": n_more_lines,
            "color_config": color_config,
        }
        
        if info_dict_symbolic is None:
            task_info['error_message'] = 'no target'
            return (False, task_info)
        
        data_info = {
            "key": fig_idx,
            "construction_cdl": plotter.construct_cdls,
            "text_cdl": info_dict_symbolic['text_cdl'],
            "image_cdl": info_dict_symbolic['image_cdl'],
            "positions": plotter.p_pos,
            "goal_cdl": info_dict_symbolic['goal_cdl'],
            "search_time": info_dict_symbolic['time'],
            "theorems": info_dict_symbolic['theorems'],
            "llm_info": {
                "key": fig_idx,
                "problem_level": info_dict_llm['problem_level'],
                "problem_text": info_dict_llm['problem_text'],
                "problem_answer": info_dict_llm['problem_answer'],
                "solution_str": info_dict_llm['solution_str'],
                "solution_dict": info_dict_llm['solution_dict'],
                "caption_str": plotter.caption_str
            }
        }
        with open(f"{info_dir}/{fig_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
            
        return (True, None)
    
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
        print('pred_base: ',  predicate_base)
        print('pred_rel: ', predicate_rel)
        return (False, task_info)

    
def generate_one_sample_with_timeout(
                          predicate_GDL, 
                          theorem_GDL, 
                          predicate_base, 
                          predicate_rel, 
                          n_more_lines,
                          color_config,
                          fig_dir, 
                          fig_idx,
                          info_dir,
                          search_cfg):
    try:
        result = func_timeout(
            # 200, 
            300,
            generate_one_sample, 
            args=(predicate_GDL, theorem_GDL, predicate_base, 
                  predicate_rel, n_more_lines, color_config, 
                  fig_dir, fig_idx, info_dir, search_cfg)
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
    # result = generate_one_sample(
    #         predicate_GDL, theorem_GDL, predicate_base, 
    #         predicate_rel, n_more_lines, color_config, 
    #         fig_dir, fig_idx, info_dir, search_cfg
    # )
    # return result



def run_task(seed, 
             task_name, 
             input_args_list,
             num_process):
    # seed, task_name, predicate_base_combs, predicate_rel_combs = task_1()
    setup_seed(seed)
    
    failure_cases_path = f"geo_synth_2/{task_name}/failure_cases.json"
    fig_dir = f"geo_synth_2/{task_name}/imgs"
    info_dir = f"geo_synth_2/{task_name}/annotations"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    print("Start Generation ...")
    
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    
    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    predicate_GDL_search = json.load(open('json/predicate_GDL_for_search.json', 'r', encoding='utf-8'))
    theorem_GDL_search = json.load(open('json/theorem_GDL_for_search.json', 'r', encoding='utf-8'))
    search_cfg = {
        "predicate_GDL": predicate_GDL_search,
        "theorem_GDL": theorem_GDL_search,
        "t_info": t_info,
        "t_freq_info": t_freq_info
    }
    
    cnt = 0
    total_iterations = len(input_args_list)
    result_info = []
    # multiprocess generate
    with Pool(processes=num_process) as pool:
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            def update(*args, **kwargs):
                pbar.update()
            for args in input_args_list:
                pred_base, pred_rel, n_more_lines, color_config = args
                result = pool.apply_async(
                    generate_one_sample, 
                    args=(predicate_GDL, theorem_GDL, pred_base, pred_rel, 
                        n_more_lines, color_config, fig_dir, cnt, info_dir, search_cfg),
                    callback=update)
                result_info.append(result)
                cnt += 1

            # get results with timeout
            results_with_timeout = []
            for i, r in enumerate(result_info):
                try:
                    result = r.get(timeout=200)
                    results_with_timeout.append(result)
                except TimeoutError:
                    print(f"Timeout for sub task {i}")
                    info = {
                        "key": cnt,
                        "pred_base": pred_base,
                        "pred_rel": pred_rel,
                        "n_more_lines": n_more_lines,
                        "color_config": color_config,
                        "error_message": "timeout"
                    }
                    results_with_timeout.append((False, info))
                except Exception as e:
                    print(f"Error occurred for sub task {i}: {e}")
                    info = {
                        "key": cnt,
                        "pred_base": pred_base,
                        "pred_rel": pred_rel,
                        "n_more_lines": n_more_lines,
                        "color_config": color_config,
                        "error_message": "timeout"
                    }
                    results_with_timeout.append((False, info))

    # save success and failure cases
    # result_info = [r.get() for r in result_info]
    failure_cases = []
    for success, symbolic_info in results_with_timeout:
        if not success:
            failure_cases.append(symbolic_info)
    
    write_json(failure_cases_path, failure_cases)
    
    print(f"Success Count: {cnt - len(failure_cases)} / {cnt}")
    print(f"Failure Count: {len(failure_cases)} / {cnt}")
    print("End for Generation.")
    

def build_input_args(pred_base_combs, 
                     pred_rel_combs, 
                     n_more_lines,
                     repeat_times):
    input_args = []
    for predicate_base in pred_base_combs:
        for predicate_rel in pred_rel_combs:
            for _ in range(repeat_times):
                color_config = np.random.choice(
                    PRESET_COLORS, p=PRESET_COLOR_PROBS)
                input_args.append(
                    (predicate_base, predicate_rel, n_more_lines, color_config)
                )
    return input_args

def task_1():
    seed = 1234
    task_name = "geosynth_ENT_1_REL_1"
    input_args_list = []
    num_process = 1
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    repeat_times=10)
    print('Num: ', len(input_args_1))

    
    input_args_list = input_args_1
    print(f'======== Task: {task_name}, Num: {len(input_args_list)} ========')
    return seed, task_name, input_args_list, num_process

def task_2():
    seed = 1234
    task_name = "geosynth_ENT_1_REL_2"
    input_args_list = []
    num_process = 6
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    repeat_times=1)
    print('Num: ', len(input_args_1))

    
    input_args_list = input_args_1
    print(f'======== Task: {task_name}, Num: {len(input_args_list)} ========')
    return seed, task_name, input_args_list, num_process

def run_task_stage_2():
    input_args_list = []
    num_process = 12
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_1 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=0,
                                    repeat_times=30)
    task_name_1 = "geo_gen_ENT_1_REL_1_L_0"
    seed_1 = 114
    print(f'Task: {task_name_1}', len(input_args_1))
    
    pred_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    pred_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    input_args_2 = build_input_args(pred_base_combs, 
                                    pred_rel_combs, 
                                    n_more_lines=1,
                                    repeat_times=30)
    task_name_2 = "geo_gen_ENT_1_REL_1_L_1"
    seed_2 = 115
    print(f'Task: {task_name_2}', len(input_args_2))
    
    
    input_args_list = [
        input_args_1, input_args_2, 
    ]
    task_name_list = [
        task_name_1, task_name_2, 
    ]
    seed_list = [
        seed_1, seed_2,
    ]
    print('Total Num: ', sum([len(args) for args in input_args_list]))
    for input_args, task_name, seed in zip(input_args_list, task_name_list, seed_list):
        print(f'======== Task: {task_name}, Num: {len(input_args)} ========')
        run_task(seed, task_name, input_args, num_process)
        
        
def main():
    # run_task(*task_1())
    # run_task(*task_2())
    # run_task()
    run_task_stage_2()

def debug_main():
    seed, task_name, input_args_list, num_process = task_1()
    setup_seed(seed)
    
    fig_dir = f"geo_synth_2/{task_name}/imgs"
    info_dir = f"geo_synth_2/{task_name}/annotations"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    print("Start Generation ...")
    
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    
    t_info = json.load(open("json/t_info_new.json", 'r', encoding='utf-8'))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    predicate_GDL_search = json.load(open('json/predicate_GDL_for_search.json', 'r', encoding='utf-8'))
    theorem_GDL_search = json.load(open('json/theorem_GDL_for_search.json', 'r', encoding='utf-8'))
    search_cfg = {
        "predicate_GDL": predicate_GDL_search,
        "theorem_GDL": theorem_GDL_search,
        "t_info": t_info,
        "t_freq_info": t_freq_info
    }
    
    cnt = 0

    # multiprocess generate
    for args in input_args_list:
        pred_base, pred_rel, n_more_lines, color_config = args
        result = generate_one_sample(
            predicate_GDL, theorem_GDL, pred_base, pred_rel, 
            n_more_lines, color_config, fig_dir, cnt, info_dir, search_cfg)
        
        print(result)
        


if __name__ == '__main__':
    main()
    # debug_main()
    