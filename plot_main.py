
import random
from multiprocessing import Pool
import os
import json
import itertools
from allocator import Allocator
from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from plotter import Plotter
from utils import PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2, setup_seed

import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import concurrent.futures



def generate_one_sample(dl, predicate_base, predicate_rel, fig_dir, fig_idx):
    try:
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            predicate_base, 
            predicate_rel, 
            n_more_lines=1
        )
        allocator = Allocator(cg.states, c_cdls, t_cdls, dl.predicate_GDL)
        allocator.allocate()
        plotter = Plotter(allocator.states)
        plotter.plot()
        fig_name = f"{fig_idx}.png"
        plotter.save_fig(fig_dir=fig_dir, fig_name=fig_name)
        info = {
            "key": fig_idx,
            "image": fig_name,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "construction_cdl": allocator.formulated_cdls['construct_cdls'],
            "text_cdl": allocator.formulated_cdls['text_cdls'], 
            "image_cdl": plotter.image_cdls
        }
        return (True, info)
    except Exception as e:
        print(f"===== Error Occured: {fig_idx} =====")
        info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "error_message": str(e)
        }
        return (False, info)
    
def run_task_with_timeout(dl, predicate_base, predicate_rel, fig_dir, fig_idx):
    try:
        result = func_timeout(
            20, 
            generate_one_sample, 
            args=(dl, predicate_base, predicate_rel, fig_dir, fig_idx)
        )
        return result
    
    except FunctionTimedOut:
        print(f"===== Time Out: {fig_idx} =====")
        info = {
            "key": fig_idx,
            "pred_base": predicate_base,
            "pred_rel": predicate_rel,
            "error_message": "timeout"
        }
        return (False, info)
    
def task_1():
    seed = 1234
    task_name = "geo_gen_ENT_1_REL_1"
    # predicate_base_combs = list(itertools.permutations(PREDICATES_ENT + PREDICATES_REL_2, 1))
    predicate_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    predicate_rel_combs = list(itertools.permutations(PREDICATES_REL, 1))
    return seed, task_name, predicate_base_combs, predicate_rel_combs

def task_2():
    seed = 1234
    task_name = "geo_gen_ENT_1_REL_2"
    predicate_base_combs = list(itertools.permutations(PREDICATES_ENT, 1))
    predicate_rel_combs = list(itertools.permutations(PREDICATES_REL, 2))
    return seed, task_name, predicate_base_combs, predicate_rel_combs

def main():
    seed, task_name, predicate_base_combs, predicate_rel_combs = task_1()
    setup_seed(seed)
    
    json_file_path = f"geo_gen/{task_name}/annotations.json"
    failure_cases_path = f"geo_gen/{task_name}/failure_cases.json"
    fig_dir = f"geo_gen/{task_name}/imgs"
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Start Generation ...")
    
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")

    cnt = 0
    total_iterations = len(predicate_base_combs) * len(predicate_rel_combs)
    result_info = []
    
    # multiprocess generate
    num_process = 12
    pool = Pool(num_process)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for predicate_base in predicate_base_combs:
            for predicate_rel in predicate_rel_combs:
                result = pool.apply_async(
                    generate_one_sample, 
                    args=(dl, predicate_base, predicate_rel, fig_dir, cnt),
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
                
    print(f"Success Count: {cnt} / {total_iterations}")
    print(f"Failure Count: {len(failure_cases)} / {total_iterations}")
    print("End for Generation. Re-generate for Failure Cases ...")
    
    # re-generate for failure cases
    failure_count = {}
    failure_dict = {}
    while True:
        if len(failure_cases) == 0:
            break
        init_info = failure_cases.pop(0)
        predicate_base = init_info['pred_base']
        predicate_rel = init_info['pred_rel']
        key = init_info["key"]
        
        if key not in failure_count:
            failure_count[key] = 0
            
        result = run_task_with_timeout(
            dl, predicate_base, predicate_rel, fig_dir, key)
        success, info = result
        
        if success:
            info_dict[key] = info
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
            else:
                failure_cases.append(info)
                failure_count[key] += 1

    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(info_dict, indent=4, ensure_ascii=False))
    with open(failure_cases_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(failure_dict, indent=4, ensure_ascii=False))
    
    print("End for Generation.")
        

if __name__ == '__main__':
    main()
    