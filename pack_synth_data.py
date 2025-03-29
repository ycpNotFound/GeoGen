import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import shutil
PREDICATES_ATTR = [
    # Attribution
    "LengthOfLine",
    "MeasureOfAngle",
    "MeasureOfArc",
    "RadiusOfCircle",
    "AreaOfTriangle",
    "DiameterOfCircle",
    "AreaOfQuadrilateral",
    "PerimeterOfTriangle",
    "LengthOfArc",
    "PerimeterOfQuadrilateral",
    "RatioOfSimilarTriangle",
    "HeightOfQuadrilateral",
    "AreaOfSector",
    "AreaOfCircle",
    "RatioOfSimilarArc",
    "RatioOfMirrorSimilarTriangle",
    "RatioOfSimilarQuadrilateral",
    "Add",
    "Mul",
    "Sub",
    "Div",
]

def replace_length_diameter(s):
    # 定义正则表达式模式，用于匹配“LengthOfLine($任意一个大写字母$) = DiameterOfCircle($任意一个大写字母$)”
    pattern_0 = r"LengthOfLine\(([A-Z]+)\) = DiameterOfCircle\(([A-Z]+)\)"
    pattern_1 = r"DiameterOfCircle\(([A-Z]+)\) = LengthOfLine\(([A-Z]+)\)"
    replace = r"\1 is diameter of circle \2"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def replace_length_radius(s):
    pattern_0 = r"LengthOfLine\(([A-Z]+)\) = RadiusOfCircle\(([A-Z]+)\)"
    pattern_1 = r"RadiusOfCircle\(([A-Z]+)\) = LengthOfLine\(([A-Z]+)\)"
    replace = r"\1 is radius of circle \2"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def measure_of_arc(s):
    pattern_0 = r"MeasureOfArc\(([A-Z]+)\)"
    pattern_1 = r"MAR_([A-Z]+)"
    replace = r"m⁀\1"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)

    return s

def measure_of_angle(s):
    pattern_0 = r"MeasureOfAngle\(([A-Z]+)\)"
    pattern_1 = r"MA_([A-Z]+)"
    replace = r"∠ \1"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def length_of_arc(s):
    pattern_0 = r"LengthOfArc\(([A-Z]+)\)"
    pattern_1 = r"LA_([A-Z]+)"
    replace = r"⁀\1"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def radius_of_circle(s):
    pattern_0 = r"RadiusOfCircle\(([A-Z])\)"
    pattern_1 = r"RC_([A-Z])"
    replace = r"r_\1"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s
    
def diameter_of_circle(s):
    pattern_0 = r"DiameterOfCircle\(([A-Z])\)"
    pattern_1 = r"DC_([A-Z])"
    replace = r"d_\1"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def perimeter_of_triangle(s):
    pattern_0 = r"PerimeterOfTriangle\(([A-Z]+)\)"
    pattern_1 = r"PT_([A-Z]+)"
    replace = r"C_\{\1\}"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def perimeter_of_quad(s):
    pattern_1 = r"PQ_([A-Z]+)"
    replace = r"C_\{\1\}"
    # s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def perimeter_of_circle(s):
    pattern_1 = r"PC_([A-Z]+)"
    replace = r"C_\{\1\}"
    s = re.sub(pattern_1, replace, s)
    return s

def area_of_triangle(s):
    pattern_0 = r"AreaOfTriangle\(([A-Z]+)\)"
    pattern_1 = r"AT_([A-Z]+)"
    replace = r"S_\{\1\}"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s



def area_of_quad(s):
    pattern_1 = r"AQ_([A-Z]+)"
    replace = r"S_\{\1\}"
    s = re.sub(pattern_1, replace, s)
    return s

def area_of_sector(s):
    pattern_1 = r"AS_([A-Z]+)"
    replace = r"S^\{sector\}_\{\1\}"
    s = re.sub(pattern_1, replace, s)
    return s

def area_of_circle(s):
    pattern_1 = r"AC_([A-Z])"
    replace = r"S_\{\1\}"
    s = re.sub(pattern_1, replace, s)
    return s

def ratio_similar_triangle(s):
    pattern_1 = r"RST_([A-Z]{3})([A-Z]{3})"
    replace = r"k_\{\1, \2\}"
    s = re.sub(pattern_1, replace, s)
    return s

def ratio_similar_quad(s):
    pattern_1 = r"RSQ_([A-Z]{4})([A-Z]{4})"
    replace = r"k_\{\1, \2\}"
    s = re.sub(pattern_1, replace, s)
    return s

def ratio_similar_arc(s):
    pattern_1 = r"RSA_([A-Z]{3})([A-Z]{3})"
    replace = r"k_\{\1, \2\}"
    s = re.sub(pattern_1, replace, s)
    return s

def ratio_mirror_similar_triangle(s):
    pattern_1 = r"RMT_([A-Z]{3})([A-Z]{3})"
    matches = re.findall(pattern_1, s)
    for match in matches:
        p1, p2 = match
        p2_reverse = p2[::-1]
        s = s.replace(f'RMT_{p1}{p2}', 
                      f"k_{{{p1},{p2_reverse}}}")

    return s

def height_of_triangle(s):
    pattern_0 = r"HeightOfTriangle\(([A-Z]+)\)"
    pattern_1 = r"HT_([A-Z]+)"
    replace = r"h_\{\1\}"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s

def height_of_quad(s):
    pattern_0 = r"HeightOfQuadrilateral\(([A-Z]+)\)"
    pattern_1 = r"HQ_([A-Z]+)"
    replace = r"h_\{\1\}"
    s = re.sub(pattern_0, replace, s)
    s = re.sub(pattern_1, replace, s)
    return s


def replace_symbols(solution):
    new_s = solution

    new_s = replace_length_diameter(new_s)
    new_s = replace_length_radius(new_s)
    new_s = measure_of_arc(new_s)
    new_s = measure_of_angle(new_s)
    new_s = length_of_arc(new_s)
    new_s = radius_of_circle(new_s)
    new_s = diameter_of_circle(new_s)
    new_s = perimeter_of_triangle(new_s)
    new_s = area_of_triangle(new_s)
    new_s = perimeter_of_quad(new_s)
    new_s = area_of_quad(new_s)
    new_s = area_of_sector(new_s)
    new_s = area_of_circle(new_s)
    new_s = ratio_similar_triangle(new_s)
    new_s = ratio_similar_quad(new_s)
    new_s = ratio_mirror_similar_triangle(new_s)
    new_s = height_of_triangle(new_s)
    new_s = height_of_quad(new_s)
    new_s = perimeter_of_circle(new_s)
    new_s = ratio_similar_arc(new_s)
    return new_s

def read_json(path):
    try:
        data = json.load(open(path, "r", encoding='utf-8'))
        return data
    except:
        return None

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)



def merge_data():
    base_dir = "D:/Desktop/GeoGen/datasets/fgo_search_train_v2"
    json_file_names = os.listdir(base_dir)
    data_reserved = {}
    print('before merge: ', len(json_file_names))
    for json_file_name in tqdm(json_file_names):
        data = read_json(os.path.join(base_dir, json_file_name))
        key = json_file_name.split('.')[0]
        # if len(data['theorems']) == 0:
        #     continue
        if data['llm_info']['solution_str'] == "Solution ":
            continue
        
        data_reserved[key] = data['llm_info']
    print('reserved: ', len(data_reserved))
    write_json(data_reserved, 'jsons/fgo_search_train_merged_v2.json')

def data_filter(solution_str, tokenizer=None):
    steps = list(range(1,16))
    final_step = 1
    for step in steps:
        if f"\n{step}. " in solution_str:
            final_step = step
    if final_step > 12:
        return False
    if tokenizer is not None:
        tokens = tokenizer.encode(solution_str)
        if len(tokens) > 1024:
            return False
    
    return True

def replace_syms_for_solution_str(solution_str):
    related_names = []
    for name in PREDICATES_ATTR:
        if f"{name}(" in solution_str:
            related_names.append(name)

    if len(related_names) != 0:
        print(related_names)
    
    solution_str = replace_symbols(solution_str)
    return solution_str

def check_circle_sstatements(solution_str, cap_str, problem_text):
    if '⊙' not in problem_text and '⊙' not in solution_str:
        return solution_str, problem_text
    pattern = r"⊙ [A-Z]{2}"
    # check problem_text
    cap_str_prefix = cap_str.split('.')[0]
    circle_str = cap_str.split('Circles:')[-1].split('.\n')[0].replace('$', '')
    circle_list = [c.strip() for c in circle_str.split(',')]
    if len(circle_list) >= 2:
        a = 1
    matches_problem = re.findall(pattern, problem_text)
    if len(matches_problem) != 0:
        if 'diameter' in cap_str_prefix:
            for match in matches_problem:
                problem_text = problem_text.replace(
                    match, f"{match.replace('⊙ ', '')} is diameter of ⊙ {circle_list[0]}"
                )
        elif 'tangent' in cap_str_prefix:
            for match in matches_problem:
                problem_text = problem_text.replace(
                    match, f"{match.replace('⊙ ', '')} is tangent to ⊙ {circle_list[0]}"
                )
        else:
            a = 1
        
    # check solution_str
    solution_prefix = solution_str.split('Solution:')[0].strip()
    solution_steps = solution_str.split('Solution:')[-1]
    
    # 1. check prefix
    matches_solution_prefix = re.findall(pattern, solution_prefix)
    if len(matches_solution_prefix) != 0:
        if 'diameter' in cap_str_prefix:
            for match in matches_solution_prefix:
                solution_prefix = solution_prefix.replace(
                    match, f"{match.replace('⊙ ', '')} is diameter of ⊙ {circle_list[0]}"
                )
            
        elif 'tangent' in cap_str_prefix:
            for match in matches_solution_prefix:
                solution_prefix = solution_prefix.replace(
                    match, f"{match.replace('⊙ ', '')} is tangent to ⊙ {circle_list[0]}"
                )
        else:
            a = 1
    
    # 2. check solution_steps
    steps = re.split(r'\n\d+\. ', solution_steps)
    steps = [s.strip() for s in steps if s.strip() != '']
    steps_correct = []
    for step in steps:
        matches_step = re.findall(pattern, step)
        if len(matches_step) != 0:
            if 'diameter' in step:
                for match in matches_step:
                    step = step.replace(
                        match, f"{match.replace('⊙ ', '')} is diameter of ⊙ {circle_list[0]}"
                    )
                steps_correct.append(step)
            elif 'tangent' in step:
                for match in matches_step:
                    step = step.replace(
                        match, f"{match.replace('⊙ ', '')} is tangent to ⊙ {circle_list[0]}"
                    )
                steps_correct.append(step)
            else:
                a = 1
        else:
            steps_correct.append(step)
    
    if solution_prefix == '':
        solution_str_correct = f"Solution:\n"
    else:
        solution_str_correct = f"{solution_prefix}\nSolution:\n"
    for i, step in enumerate(steps_correct):
        solution_str_correct += f"{i+1}. {step}\n"
                
    return solution_str_correct, problem_text
   


def filter_data():
    # merge_data()
    # data_path = 'F:/GeoDatasets/GeoSynth/Stage2_v3/geo_synth_stage_2_v3.json'
    data_path = 'F:/Dataset/GeoSynth/Stage2_v4/geo_synth_stage_2_v5.json'
    data = read_json(data_path)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-72B-Instruct')
    data_reserved = {}
    print('before filter: ', len(data))

    level_dict = {}

    for key, value in tqdm(data.items()):
        solution_str = value['solution_str']
        level = value['problem_level']
        # solution_str = formulate_solution_str(value['solution_dict'])
        solution_str = replace_syms_for_solution_str(solution_str)
        solution_str,  problem_text = check_circle_sstatements(
            solution_str, 
            value['caption_str'],
            value['problem_text'],
        )
        res = data_filter(tokenizer, solution_str)
        if res:
            data_reserved[key] = value
            data_reserved[key]['solution_str'] = solution_str

    print('after filter: ', len(data_reserved))
    # resort
    # keys like 1, 1_0, 1_1, .. 2, 2_0, 2_1, .., sort by this pattern
    def index_func(key):
        if '_' not in key:
            return (int(key), 0)
        else:
            return (int(key.split('_')[0]), int(key.split('_')[1]) + 1)

    # data_reserved = dict(sorted(data_reserved.items(), key=lambda x: index_func(x[0])))

    write_json(data_reserved, 'geo_synth_2/geo_synth_stage_2_v5_filtered.json')
    # return data_reserved

def process_item(item, cnt, source_dir):
    k, v = item
    cap_str_prefix = v['caption_str'].split('.')[0]
    if 'v4' in source_dir or ',' in cap_str_prefix:
        image_source = f'{source_dir}/{k}.png'
        image_target = f'F:/Dataset/GeoSynth/Stage2_v4/final/imgs/{cnt}.png'
        # if not os.path.exists(image_target):
        #     shutil.copy(image_source, image_target)
            
        return True, str(k), str(cnt)
        
    return False, None, None

def process_item_batch(input_args, source_dir):
    res_lst = []
    for item, target_k in tqdm(input_args):
        res = process_item(item, target_k, source_dir)
        res_lst.append(res)
    return res_lst

def filter_data_v2():
    import shutil
    from multiprocessing import Pool
    from functools import partial
    # merge_data()
    # data_path = 'F:/GeoDatasets/GeoSynth/Stage2_v3/geo_synth_stage_2_v3.json'
    data_path = 'F:/Dataset/GeoSynth/Stage2_v4/geo_synth_stage_2_v5.json'
    data_path_2 = 'F:/Dataset/GeoSynth/Stage2_v3/geo_synth_stage_2_v3.json'
    data = read_json(data_path)
    data_2 = read_json(data_path_2)
    final_dir = 'F:/Dataset/GeoSynth/Stage2_v4/final/imgs'
    os.makedirs(final_dir, exist_ok=True)
    
    cnt = 0
    total_data = {}
    n_worker = 2
    # final_key to origin_key
    source_key_map = {}
    key_list_1 = [str(x) for x in range(len(data))]
    input_args_1 = [
        (item, k) for item, k in zip(data.items(), key_list_1)
    ]
    for arg in input_args_1:
        final_key = arg[1]
        origin_key = arg[0][0]
        source_key_map[final_key] = origin_key
        
    batch_size = len(input_args_1) // n_worker
    batches = []
    for i in range(n_worker):
        batch = input_args_1[i*batch_size:(i+1)*batch_size]
        batches.append(batch)

    with Pool(processes=n_worker) as pool:
        batch_results = pool.map(
            partial(
                process_item_batch,
                source_dir='F:/Dataset/GeoSynth/Stage2_v4/imgs'
                ),
                batches
            )
    total_results = []
    for batch_result in batch_results:
        total_results += batch_result
    for saved, source_k, target_k in total_results:
        total_data[target_k] = data[source_k]

    print(len(total_data))
    print(len(os.listdir('F:/Dataset/GeoSynth/Stage2_v4/final/imgs')))

    # for data 2
    key_list_2 = [str(x) for x in range(len(data), len(data) + len(data_2))]
    input_args_2 = [
        (item, k) for item, k in zip(data_2.items(), key_list_2)
    ]
    for arg in input_args_2:
        final_key = arg[1]
        origin_key = arg[0][0]
        source_key_map[final_key] = origin_key
        
    batch_size = len(input_args_2) // n_worker
    
    write_json(source_key_map, 'source_k_map.json')
    batches = []
    for i in range(n_worker):
        batch = input_args_2[i::n_worker]
        batches.append(batch)
    print(len(batches))
    with Pool(processes=n_worker) as pool:
        batch_results = pool.map(
            partial(
                process_item_batch,
                source_dir='F:/Dataset/GeoSynth/Stage2_v3/imgs'
                ),
                batches
            )
    total_results = []
    for batch_result in batch_results:
        total_results += batch_result
    for saved, source_k, target_k in total_results:
        if saved:
            total_data[target_k] = data_2[source_k]



    # for k, v in tqdm(data_2.items()):
    #     cap_str_prefix = v['caption_str'].split('.')[0]
    #     if ',' in cap_str_prefix:
    #         image_source = f'F:/Dataset/GeoSynth/Stage2_v3/imgs/{k}.png'
    #         image_target = f'F:/Dataset/GeoSynth/Stage2_v4/final/imgs/{cnt}.png'
    #         shutil.copy(image_source, image_target)
    #         total_data[str(cnt)] = v
    #         cnt += 1
    # key_list = [str(x) for x in range(len(data), len(data) + len(data_2))]
    # input_args = [
    #     (item, k) for item, k in zip(data_2.items(), key_list)
    # ]
    # batch_size = len(input_args) // 6
    # batches = []
    # for i in range(6):
    #     batch = input_args[i*batch_size:(i+1)*batch_size]
    #     batches.append(batch)
    # assert len(key_list) == len(input_args)
    # with Pool(processes=6) as pool:
    #     with tqdm(total=len(input_args)) as pbar:
    #         for res in pool.starmap(process_item, input_args):
    #             is_save, source_k, target_k = res
    #             if is_save:
    #                 total_data[target_k] = data_2[source_k]
    #             pbar.update(1)


            
    print(len(total_data))
    print(len(os.listdir('F:/Dataset/GeoSynth/Stage2_v4/final/imgs')))

    data = total_data
    
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-72B-Instruct')
    data_reserved = {}
    print('before filter: ', len(data))

    level_dict = {}

    for key, value in tqdm(data.items()):
        solution_str = value['solution_str']
        level = value['problem_level']
        # solution_str = formulate_solution_str(value['solution_dict'])
        solution_str = replace_syms_for_solution_str(solution_str)
        solution_str,  problem_text = check_circle_sstatements(
            solution_str, 
            value['caption_str'],
            value['problem_text'],
        )
        res = data_filter(solution_str)
        if res:
            data_reserved[key] = value
            data_reserved[key]['solution_str'] = solution_str

    print('after filter: ', len(data_reserved))

    write_json(data_reserved, 'F:/Dataset/GeoSynth/Stage2_v4/final/geo_synth_stage_2_v6_filtered.json')
    # return data_reserved

def simplify_solution():
    # data_path = 'geo_synth_2/geo_synth_stage_2_v6_filtered.json'
    data_path = 'F:/Dataset/GeoSynth/Stage2_v4/final/geo_synth_stage_2_v6_filtered.json'
    data = read_json(data_path)

    data_reserved = {}
    for key, value in tqdm(data.items()):
        solution_str = value['solution_str'].strip()
        solution_str = solution_str.replace('Solution: ', '')
        steps = re.split(r'\n\d+\. ', solution_str)
        steps = [s.strip() for s in steps if s != '']
        steps_simple = []
        for step in steps:
            if '<by>' in step and '<because>' in step:
                matches = re.findall(r"<by> \S+,", step)
                if matches:
                    t_str = matches[0]
                    step = step.replace(t_str, '')
                    
            step = step.strip()
            steps_simple.append(step)

        solution_simple = ""
        for i, step in enumerate(steps_simple):
            solution_simple += f"{i+1}. {step}\n"
        solution_simple = solution_simple.strip()
        data_reserved[key] = value
        data_reserved[key]['solution_str'] = solution_simple

    f_name = 'F:/Dataset/GeoSynth/Stage2_v4/final/geo_synth_stage_2_v6_filtere_sim.json'
    write_json(data_reserved, f_name)



def get_regen_keys():
    data_path = 'jsons/fgo_search_train_merged.json'
    data = read_json(data_path)
    solutions = [item['solution_str'] for item in data.values()]

    def check(query, solutions):
        for solution in solutions:
            if query in solution:
                return True
        return False
    
    q_list = []
    for i in range(100, 180):
        q = list(str(i))
        q = ''.join([q[-1], q[0], q[1]])
        q_list.append(q)
    # print(q_list)
    regen_keys = []
    for k, item in tqdm(data.items()):
        for q in q_list:
            if q in item['solution_str']:
                regen_keys.append(k.split('_')[0])
                break
    regen_keys = list(set(regen_keys))
    write_json(regen_keys, 'jsons/regen_keys_v2.json')

if __name__ == "__main__":
    # merge_data()
    # filter_data()
    filter_data_v2()
    # merge_data_pgps()
    # filter_data_pgps()
    simplify_solution()
    # for q in q_list:
    #     if check(q, solutions):
    #         print(q)
