
import itertools
import json
import os
import traceback
import random
import time
import copy
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from allocator import Allocator
from generator import ClauseGenerator
from plotter import Plotter
from graph import (ConditionGraph, ConditionNode, draw_graph, topological_sort,
                   topological_sort_bfs)
from utils.preset import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,
                          PRESET_COLOR_PROBS, PRESET_COLORS)
from utils.tools import identify_image, setup_seed, write_json
from solver import FormalGeoSolver
from target_finder import TargetFinder

class Expander(TargetFinder):
    def __init__(self, 
                 predicate_GDL,
                 theorem_GDL,
                 t_info,
                 t_freq_info,
                 text_cdls,
                 constr_cdls,
                 image_cdls,
                 debug=False):
        allocater_states = {
            'p_pos': None,
            'lines': None,
            'circles': None,
            'points_on_circle': None,
            'clauses': None,
        }
        super().__init__(
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
            predicate_num=2,
            debug=debug
        )

    def find_targets_and_solutions(self, condition_graph: ConditionGraph, strict=True):
        conditions_to_sample = self.get_conditions_to_sample()
        # filter 1
        new_targets = self.targets_filter_1(
            conditions_to_sample,
            self.solver.problem.condition.value_of_sym
        )
        # filter 2, if has too many targets, filter by more strict rules
        new_targets = self.targets_filter_2(new_targets, strict=strict)
            
        # find solution / theorems for each target
        theorems_for_targets = {}
        solution_for_targets = {}
        solution_dict_for_targets = {}
        # draw_graph(condition_graph, 'test', new_targets[11], img_dir='imgs_test')
        for target in new_targets:
            (
                solution_str,
                solution_formal_dict,
                theorems, 
                sub_nodes,
                too_complex_flag
            ) = self.find_solution_for_target(
                self.solver.problem,
                condition_graph, 
                target, 
                self.natural_template,
                self.solver.parsed_theorem_GDL
            )
            if too_complex_flag:
                continue

            theorems_for_targets[target] = [item['theorem'] for item in solution_formal_dict.values() if item['theorem'] is not None]
            solution_for_targets[target] = solution_str
            solution_dict_for_targets[target] = solution_formal_dict

            # level = sum([1 if item['theorem'] is not None else 0 for item in solution_formal_dict.values()])

        if self.debug:
            _solution_for_targets = {
                str(k): v.split('\n') for k, v in
                solution_for_targets.items()
            }
            with open('json/solution_test.json', 'w', encoding='utf-8') as f:
                json.dump(_solution_for_targets, f, indent=4, ensure_ascii=False)
        
        # filter 3, sort by rules
        new_targets, idx_for_targets = self.targets_filter_3(
            new_targets, theorems_for_targets, self.max_depth
        )
        # formulate into groups by target type
        targets_dict = self.targets_into_groups(new_targets)
        types_to_choose = [k for k in targets_dict if len(targets_dict[k]) != 0]
        if len(types_to_choose) == 0:
            return None
        
        # re-sort
        available_targets = []
        for target_type in types_to_choose:
            for target in targets_dict[target_type]:
                available_targets.append((target, target_type))
        
        # choose all top-10 targets
        result_list = []
        chosen_targets = available_targets[:10]
        for chosen_target, target_type in chosen_targets:    
            chosen_thoerems = theorems_for_targets[chosen_target]
            chosen_solution = solution_for_targets[chosen_target]
            chosen_solution_dict = solution_dict_for_targets[chosen_target]
            # problem_level = depth_for_targets[chosen_target]
            problem_level = len(chosen_thoerems)
            result_list.append(
                (
                    target_type,
                    chosen_target,
                    problem_level,
                    chosen_solution,
                    chosen_solution_dict,
                    chosen_thoerems
                )
            )
        return result_list

    def formulate(self):
        start_time = time.time()
        self.solver.init_search(self.problem_CDL)
        self.solver.search()
        # self.solver.bfs_search()
        cost_time = time.time() - start_time
        if self.debug:
            print(f"Search took {cost_time:.5f} seconds. ")
        # construct condition graph
        condition_graph = ConditionGraph(self.solver.problem.condition.items)
        condition_graph.construct_graph()
        
        # find potential geo relation ()
        result_list = self.find_targets_and_solutions(condition_graph, strict=True)

        if result_list is None or len(result_list) == 0:
            return []
        
        info_list = []
        for args in result_list:
            target_type, target, problem_level, solution_str, solution_dict, theorems = args
            if problem_level == 1:
                continue
            # create question and solution for this geo relation
            (
                conclusion,
                add_cdls,
                add_conditions,
                target_value, 
                target_cdl, 
                problem_text,
                problem_text_type
            ) = self.create_question(target, problem_text_type='image_based')

            self.text_cdls += add_cdls
            self.image_cdls += add_cdls

            if len(add_conditions) != 0:
                solution_str += f"\n<because> {', '.join(add_conditions)}, <therefore> {conclusion}."
            else:
                solution_str += f"\n<therefore> {conclusion}."

            info_dict_for_symbolic = {
                "problem_level": problem_level,
                "problem_text_en": problem_text,
                "construction_cdl": self.constr_cdls,
                "text_cdl": self.text_cdls,
                "image_cdl": self.image_cdls,
                "goal_cdl": target_cdl,
                "problem_answer": str(target_value),
                "theorems": theorems,
                "time": round(cost_time, 3),
            }
            info_dict_for_llm = {
                "problem_type": target_type,
                "problem_level": problem_level,
                "problem_text": problem_text,
                "problem_text_type": problem_text_type,
                "problem_answer": str(target_value),
                "solution_str": solution_str,
                "solution_dict": solution_dict
            }
            info_list.append(
                (
                    info_dict_for_symbolic,
                    info_dict_for_llm
                )
            )
            
        return info_list

def expand_one_sample(
    predicate_GDL, 
    theorem_GDL,
    problem_CDL,
    data_info,
    debug=False,
    t_info=None,
    t_freq_info=None,
    save_dir='datasets/fgo_train_search',
):
    expander = Expander(
        predicate_GDL,
        theorem_GDL,
        t_info,
        t_freq_info,
        text_cdls=problem_CDL['text_cdl'],
        constr_cdls=problem_CDL['construction_cdl'],
        image_cdls=problem_CDL['image_cdl'],
        debug=debug
    )

    result_list = expander.formulate()
    
    if len(result_list) == 0:
        return False

    cnt = 0
    for info_dict_symbolic, info_dict_llm in result_list:
        cnt += 1
        data_info = {
            "key": problem_CDL['problem_id'],
            "source": data_info['source'],
            "construction_cdl": info_dict_symbolic['construction_cdl'],
            "text_cdl": info_dict_symbolic['text_cdl'],
            "image_cdl": info_dict_symbolic['image_cdl'],
            "goal_cdl": info_dict_symbolic['goal_cdl'],
            "search_time": info_dict_symbolic['time'],
            "theorems": info_dict_symbolic['theorems'],
            "llm_info": {
                "key": problem_CDL['problem_id'],
                "problem_level": info_dict_llm['problem_level'],
                "problem_text": info_dict_llm['problem_text'],
                "problem_answer": info_dict_llm['problem_answer'],
                "solution_str": info_dict_llm['solution_str'],
                "solution_dict": info_dict_llm['solution_dict'],
            }
        }
        with open(f"{save_dir}/{problem_CDL['problem_id']}_{cnt}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
        
    return True

def expand_main():
    data_path = "datasets/processed_data/fgo_train.json"
    save_dir = 'datasets/fgo_train_search'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = json.load(open(data_path, 'r', encoding='utf-8'))


    t_info = json.load(open("json/t_info_new.json"))
    t_names = sorted(t_info, reverse=True, key=lambda k: t_info[k][-1])
    t_freq_info = {k: t_info[k][-1] for k in t_names}
    predicate_GDL_search = json.load(open('json/predicate_GDL_for_search.json', 'r'))
    theorem_GDL_search = json.load(open('json/theorem_GDL_for_search.json', 'r'))
    
    for k, v in tqdm(data.items()):
        if int(k) < 11:
            continue
        problem_CDL = {
            "problem_id": v['problem_id'],
            "construction_cdl": v['construction_cdl'],
            "text_cdl": v['text_cdl'],
            "image_cdl": v['image_cdl'],
            "goal_cdl": v['goal_cdl'],
            "problem_answer": v['problem_answer'],
            "point_positions": None,
            "line_instances": None,
            "circle_instances": None
        }
        print(f'==================={k}===================')
        res = expand_one_sample(
            predicate_GDL_search,
            theorem_GDL_search,
            problem_CDL,
            debug=True,
            t_info=t_info,
            t_freq_info=t_freq_info,
            data_info=v,
            save_dir=save_dir,
        )
        
        a = 1
        
if __name__ == "__main__":
    expand_main()

