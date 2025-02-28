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


    