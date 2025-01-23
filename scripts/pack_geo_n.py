import json
import os
import random
from tqdm import tqdm


def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def write_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def process_geo3k_data():
    # 找到没有标字母的图像，往solution里加条件
    return


def get_source_problem(key):
    base_dir = 'datasets/formalgeo7k/problems'
    
    data = json.load(open(f"{base_dir}/{key}.json", 'r', encoding='utf-8'))
    source = data['source']
    
    
def get_source_data(source_key):
    geo3k_dir = 'D:/Desktop/资源/几何答题/InterGPS-main/data/geometry3k'
    geoqa_dir = 'D:/Desktop/资源/几何答题/GeoQA/GeoQA3/json'
    if 'Geometry3k' in source_key:
        idx = int(source_key.split('-')[-1])
        if idx <= 2100:
            split = 'train'
        elif idx <= 2400 and idx > 2100:
            split = 'val'
        else:
            split = 'test'
        geo3k_path = f"{geo3k_dir}/{split}/{idx}/data.json"
        if os.path.exists(geo3k_path):
            geo3k_data = json.load(open(geo3k_path, 'r', encoding='utf-8'))
            return geo3k_data
        else:
            raise KeyError(geo3k_path)
    elif 'GeoQA':
        if 'Angle' in source_key:
            idx = int(source_key.split('-')[-1])
        else:
            idx = int(source_key.split('-')[-1])
          

def formulate_info(source_key, sym_json, llm_json):
    source_data = get_source_data(source_key)
    if 
    
    # move image to new dir
    
    # check if there's unmarked points, 
    # if true, add hypothesis points at the start of solution
    
    
    # random choose solution type: 
    # answer directly (synthetic) or choose from ABCD (origin)

    
    # random choose problem text type: 
    # Rich in condition (synthetic) or directly command answer (origin)
    
    
    ori_problem_flag = random.choice([True, False])
    if ori_problem_flag:
        problem_text = 
    info = {
        "image": image_path,
        "source": source_key,
        "level": sym_json['llm_info']['problem_level'],
        "problem_text": problem_text,
        "solution": solution
    }
    
    
"""
GeoSN Dataset, consist of:
- FormalGeo7k
    - GeoQA
    - Geometry3K
- UniGeo-porve
- PGPS9K (todo)
with natural language solution

annotation format: {
    "image": image_id, move to new dir and name: "{ori_set}_{key}.png"
    "source": ori_set, origin_key,
    "level": problem_level,
    "problem_text": text,
    "solution": solution to problem
}
None: 
1. if origin problem has un-noted points, then add hyposis points in the beginning of solution
2. if origin problem has choices, random append choices at the end
"""

def process_data():
    global_cnt = 0
    total_data = {}
    
    # 处理所有的formalgeo data，包括原始数据，expand数据
    json_dir_llm = 'datasets/processed_data/fgo_train_chat'
    json_dir_sym = 'datasets/processed_data/fgo_train'
    json_dir_expand = 'datasets/processed_data/fgo_train_expand'
    print(f'train total: {len(os.listdir(json_dir_llm))}')
    for name in os.listdir(json_dir_llm):
        key = name.split('.')[0]
        llm_json = read_json(os.path.join(json_dir_llm, name))
        sym_json = read_json(os.path.join(json_dir_sym, name))
        
        if len(llm_json) == 0:
            continue
        
        source_key = sym_json['source']
        
        # origin problem
        total_data[global_cnt] = formulate_info(
            source_key, 
            sym_json, 
            llm_json[key]
        )
        global_cnt += 1
        
        expanded_keys = [k for k in llm_json if str(k) != key]
        for i, expand_k in enumerate(expanded_keys):
            expand_sym_json = read_json(os.path.join(
                json_dir_expand, f"{expand_k}.json"
            ))
            total_data[global_cnt] = formulate_info(
                f"{source_key}-expand-{i}", 
                expand_sym_json, 
                llm_json[expand_k]
            )
            global_cnt += 1
        
    