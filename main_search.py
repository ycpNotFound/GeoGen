
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
from utils.tools import identify_image, setup_seed


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
            debug=True
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
        return (False, info, None)
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
    t_freq_info = json.load(open("json/theorem_freq.json", 'r', encoding='utf-8'))
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
    pool = Pool(num_process)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for args in input_args_list:
            pred_base, pred_rel, n_more_lines, color_config = args
            result = pool.apply_async(
                generate_one_sample_with_timeout, 
                args=(predicate_GDL, theorem_GDL, pred_base, pred_rel, 
                      n_more_lines, color_config, fig_dir, cnt, info_dir, search_cfg),
                callback=update)
            result_info.append(result)
            cnt += 1

        for r in result_info:
            r.wait()

    # save success and failure cases
    result_info = [r.get() for r in result_info]
    failure_cases = []

    for success, symbolic_info in result_info:
        if success:
            pass
        else:
            failure_cases.append(symbolic_info)
                
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
                predicate_GDL, theorem_GDL, pred_base, pred_rel, 
                n_more_lines, color_config, fig_dir, cnt, info_dir, search_cfg
            )
            
            success, symbolic_info = result
            if success:
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
                    failure_cases.append(symbolic_info)
                    failure_count[key] += 1


    with open(failure_cases_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(failure_dict, indent=4, ensure_ascii=False))
    
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

def main():
    run_task(*task_1())
    # run_task()

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
    