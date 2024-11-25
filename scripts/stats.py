import json
import re
from collections import Counter

def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def load_theorems_and_predicates():
    path_1 = "datasets/predicates.txt"
    path_2 = "datasets/theorems.txt"
    
    with open(path_1, 'r') as f1:
        preds = f1.readlines()
        preds = [s.strip() for s in preds]
        
    with open(path_2, 'r') as f2:
        theos = f2.readlines()
        theos = [s.strip() for s in theos]
        
    return preds, theos

def stats_for_formalgeo():
    split = "train"
    data_path = f"datasets/processed_data/fgo_{split}.json"
    data = read_json(data_path)
    preds, theos = load_theorems_and_predicates()
    
    cdl_list, theo_list = [], []
    for key, value in data.items():
        cdl_list_i = value['construction_cdl'] + value['text_cdl']
        
            
        cdl_list_1 = [s.split('(')[0] for s in cdl_list_i]
        cdl_list_2 = []
        for s in cdl_list_i:
            if 'Equal' in s:
                s = s.split('Equal(')[-1][:-1]
                s = s.split('(')[0]
                if s[0].isupper():
                    cdl_list_2.append(s)
        
        cdl_list_i = cdl_list_1 + cdl_list_2
        theo_list_i = value['theorem_seqs']
        theo_list_i = [s.split('(')[0] for s in theo_list_i]
        
        cdl_list += cdl_list_i
        theo_list += theo_list_i
    
    cdl_count = Counter(cdl_list)
    theo_count = Counter(theo_list)
    
    preds_not_used = set(preds) - set(list(cdl_count.keys()))
    print(f"{'-'*10} Predicats not used : {'-'*10}")
    for p in preds_not_used:
        print(p)
        
    theo_not_used = set(theos) - set(list(theo_count.keys()))
    print(f"{'-'*10} Theorems not used: {'-'*10}")
    for t in theo_not_used:
        print(t)
        
    pred_sorted = sorted(list(cdl_count.keys()), key=lambda x: cdl_count[x], reverse=True)
    print(f"{'-'*10} Predicats Stats: {'-'*10}")
    for p in pred_sorted:
        print(f"{cdl_count[p]} - {p}")
        
    theo_sorted = sorted(list(theo_count.keys()), key=lambda x: theo_count[x], reverse=True)
    print(f"{'-'*10} Theorem Stats: {'-'*10}")
    for t in theo_sorted:
        print(f"{theo_count[t]} - {t}")
    
    cdl_dict = dict(sorted(cdl_count.items(), key=lambda item: item[1], reverse=True))
    for key in preds_not_used:
        cdl_dict[key] = 0
        
    theo_dict = dict(sorted(theo_count.items(), key=lambda item: item[1], reverse=True))
    for key in theo_not_used:
        theo_dict[key] = 0
    with open('json/predicate_freq.json', 'w', encoding='utf-8') as f:
        json.dump(cdl_dict, f, indent=4, ensure_ascii=False)
    with open('json/theorem_freq.json', 'w', encoding='utf-8') as f:
        json.dump(theo_dict, f, indent=4, ensure_ascii=False)
        
    return

if __name__ == '__main__':
    stats_for_formalgeo()