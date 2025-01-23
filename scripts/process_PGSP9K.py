import json
import os
import re
import itertools
from tqdm import tqdm

def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def write_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def PGPS9K_to_formal_clause(statement):
    clauses = []
    if 'line' in statement:
        if 'lieson' in statement: # line r lieson D B C
            pattern = r"line (\w) lieson ([A-Z](?: [A-Z])*)"
            match = re.search(pattern, statement)
            if match:
                points = match.group(2).split()
                assert len(points) >= 3
                for sub_ps in itertools.combinations(points, 3):
                    clauses.append(f"Collinear({''.join(sub_ps)})")
        elif '\\perp' in statement:
            pass

def read_pgps():
    json_path = 'D:/Desktop/资源/几何答题/PGPS9K_all/PGPS9K/train.json'
    data = read_json(json_path)
    for k, v in data.items():
        statements = v['parsing_stru_seqs'] + v['parsing_sem_seqs']
        