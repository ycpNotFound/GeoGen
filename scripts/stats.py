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

def is_theorem_equal(theorems_1, theorems_2):
    t_names_1 = [t.split('(')[0] for t in theorems_1]
    t_names_2 = [t.split('(')[0] for t in theorems_2]

    if len(t_names_1) == len(t_names_2):
        if all([t1 == t2 for t1, t2 in zip(t_names_1, t_names_2)]):
            return True
    
    return False


def stats_for_synth_2(dir_id=1):
    import os
    
    base_dir = f'geo_synth_2/geosynth_ENT_1_REL_{dir_id}/annotations'
    files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
    file_keys = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]

    # find missing keys
    max_num = max(file_keys)
    missing_keys = []
    for i in range(max_num):
        if i not in file_keys:
            missing_keys.append(i)
    print(f"Exist keys: {len(file_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    
    # find distribution of theorems and difficulty
    data_list = [
        json.load(open(f"{base_dir}/{i}.json", 'r', encoding='utf-8'))
        for i in file_keys
    ]
    data_info_list = []

    for key, data in zip(file_keys, data_list):
        pred_ent = data['text_cdl'][0].split('(')[0]
        pred_rel = data['text_cdl'][1].split('(')[0]

        info = {
            "pred_ent": pred_ent,
            "pred_rel": pred_rel,
            "level": data['llm_info']['problem_level'],
            "time": data['search_time'],
            "theorems": data['theorems']
        }
        if info['level'] > 20:
            print(f"geo_synth_2/geosynth_ENT_1_REL_{dir_id}/annotations/{key}.json")
            print(f"geo_synth_2/geosynth_ENT_1_REL_{dir_id}/imgs/{key}.png")
        data_info_list.append(info)
    
    
    # stats level, time for each predicate
    info_for_predicate_ent = {}
    info_for_predicate_rel = {}
    for info in data_info_list:
        if info['pred_ent'] not in info_for_predicate_ent:
            info_for_predicate_ent[info['pred_ent']] = []
        else:
            info_for_predicate_ent[info['pred_ent']].append(info)
        if info['pred_rel'] not in info_for_predicate_rel:
            info_for_predicate_rel[info['pred_rel']] = []
        else:
            info_for_predicate_rel[info['pred_rel']].append(info)
            
    for pred, info_list in info_for_predicate_ent.items():
        # stats level and time for each predicate
        level_count = Counter([info['level'] for info in info_list])
        level_count = {i: level_count[i] for i in sorted(level_count)}
        time_list = [info['time'] for info in info_list]
        
        
        # stats theorems and repeat rate of theorems
        theorems_seqs = [info['theorems'] for info in info_list]
        theorems_seqs_unique = []
        
        for i, theorems_i in enumerate(theorems_seqs):
            for theorems_j in theorems_seqs_unique:
                if is_theorem_equal(theorems_i, theorems_j):
                    break
            else:
                theorems_seqs_unique.append(theorems_i)
        
        repeat_rate = len(theorems_seqs_unique) / len(theorems_seqs) 
        
        print(f"{'-'*10} {pred} {'-'*10}")
        print(f"Level: {level_count}")
        print(f"Time: Avg: {(sum(time_list) / len(time_list)):.3f}, Max: {max(time_list):.3f}, Min: {min(time_list):.3f}")
        print(f"Repeat rate: {repeat_rate}")
    
    print('=================================')
    for pred, info_list in info_for_predicate_rel.items():
        # stats level and time for each predicate
        level_count = Counter([info['level'] for info in info_list])
        level_count = {i: level_count[i] for i in sorted(level_count)}
        time_list = [info['time'] for info in info_list]
        
        
        # stats theorems and repeat rate of theorems
        theorems_seqs = [info['theorems'] for info in info_list]
        theorems_seqs_unique = []
        
        for i, theorems_i in enumerate(theorems_seqs):
            for theorems_j in theorems_seqs_unique:
                if is_theorem_equal(theorems_i, theorems_j):
                    break
            else:
                theorems_seqs_unique.append(theorems_i)
        
        repeat_rate = len(theorems_seqs_unique) / len(theorems_seqs) 
        
        print(f"{'-'*10} {pred} {'-'*10}")
        print(f"Level: {level_count}")
        print(f"Time: Avg: {(sum(time_list) / len(time_list)):.3f}, Max: {max(time_list):.3f}, Min: {min(time_list):.3f}")
        print(f"Repeat rate: {repeat_rate}")
        
    # plot time distribution
    from matplotlib import pyplot as plt
    
    img_dir = 'imgs_test'
    times = [info['time'] for info in data_info_list]
    plt.hist(times, bins=50)
    plt.savefig(os.path.join(img_dir, f"time_dist_{dir_id}.png"))
    plt.clf()
    
    levels = Counter([info['level'] for info in data_info_list])
    x = list(levels.keys())
    y = [levels[i] for i in x]
    plt.bar(x, y)
    plt.savefig(os.path.join(img_dir, f"level_dist_{dir_id}.png"))
    
    # all repeat rate
    theorems_seqs = [info['theorems'] for info in data_info_list]
    theorems_seqs_unique = []
    
    for i, theorems_i in enumerate(theorems_seqs):
        for theorems_j in theorems_seqs_unique:
            if is_theorem_equal(theorems_i, theorems_j):
                break
        else:
            theorems_seqs_unique.append(theorems_i)
    repeat_rate = len(theorems_seqs_unique) / len(theorems_seqs) 
    print(f'All Repeat rate: {repeat_rate:.3f}' )
    print('=================================')
    
    # predicate sort
    keys, freqs = [], []
    for pred, info_list in info_for_predicate_rel.items():
        keys.append(pred)
        freqs.append(len(info_list))
        
    keys, freqs = zip(*sorted(zip(keys, freqs), key=lambda x: x[1], reverse=True))
    for k, f in zip(keys, freqs):
        print(f'{k}: {f}')
    
    print('=================================')
    # theorem distribution
    seen_theorems = {}
    for info in data_info_list:
        for theorem in info['theorems']:
            t = theorem.split('(')[0]
            if t not in seen_theorems:
                seen_theorems[t] = 1
            else:
                seen_theorems[t] += 1

    total_theorems = json.load(open('json/t_info_new.json', 'r', encoding='utf-8'))
    total_theorems = [t for t in total_theorems if total_theorems[t][1] != 0]
    
    print('Total: ', len(total_theorems))
    print('Used: ', len(seen_theorems))

    seen_dict = {}
    unseen_dict = {}
    for t in total_theorems:
        if t in seen_theorems:
            seen_dict[t] = seen_theorems[t]
        else:
            unseen_dict[t] = ""
    seen_dict = {k: seen_dict[k] for k in 
                sorted(seen_dict, key=lambda x: seen_dict[x], reverse=True)}
    with open('json/seen_theorems.json', 'w', encoding='utf-8') as f:
        json.dump(seen_dict, f, ensure_ascii=False, indent=4)
    with open('json/unseen_theorems.json', 'w', encoding='utf-8') as f:
        json.dump(unseen_dict, f, ensure_ascii=False, indent=4)
        
    # special cases
    
    return



if __name__ == '__main__':
    # stats_for_formalgeo()
    stats_for_synth_2(dir_id=2)