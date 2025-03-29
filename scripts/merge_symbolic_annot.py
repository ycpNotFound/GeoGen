import json
import os
from tqdm import tqdm


def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def write_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def merge_main():
    symbolic_dict = {}
    pgps_symbolic_path = 'D:/Desktop/资源/几何答题/UniAll/jsons/pgps9k/train_formal_img.json'
    pgps_symbolic_data = read_json(pgps_symbolic_path)
    for k, v in pgps_symbolic_data.items():
        pass


if __name__ == '__main__':
    merge_main()
