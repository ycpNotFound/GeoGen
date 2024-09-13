import json
import os
import pickle

from tqdm import tqdm


def read_json(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    return data

def read_pickle(pk_path):
    data = pickle.load(open(pk_path, 'rb'))
    return data

def read_jsons(json_dir):
    files = os.listdir(json_dir)
    files = [s for s in files if s.endswith('.json')]
    datas = []
    for file in files:
        data_i = read_json(os.path.join(json_dir, file))
        datas.append(data_i)
        
    return datas

def save_json(data, json_path):
    json_dir = os.path.dirname(json_path)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

def stats_source():
    json_dir = "datasets/formalgeo7k/problems"
    print('Origin Fgo Len: ', len(os.listdir(json_dir)))
    data_list = read_jsons(json_dir)

    
    dataset_dict = {
        "Geometry3k": [],
        "GeoQA": [],
    }
    # forward and backward search for keys
    formal2geo_dict = {}
    geo2formal_dict = {}
    formal_geo_data = {}
    for i, data in enumerate(data_list):
        problem_id = data['problem_id']
        source = data['source'].split('-')[0]
        dataset_dict[source].append(data['source'])
        formal_geo_data[problem_id] = data
        formal2geo_dict[problem_id] = data['source']
        geo2formal_dict[data['source']] = problem_id
        
    for key, value in dataset_dict.items():
        value.sort()
    
    # save_json(dataset_dict, 'datasets/processed_data/data_keys.json')
    print('Fgo from Geo3k: ', len(dataset_dict['Geometry3k']))
    print('Fgo from GeoQA: ', len(dataset_dict['GeoQA']))
    print('')

    valid_geoqa, test_geoqa = stats_for_geoqa()
    valid_geo3k, test_geo3k = stats_for_geometry3k()
    
    # split and merge
    formal_train_keys, formal_val_keys, formal_test_keys = [], [], []
    
    for prefix_idx in dataset_dict['Geometry3k']:
        idx = prefix_idx.replace('Geometry3k-', '')
        if idx in valid_geo3k:
            formal_val_keys.append(geo2formal_dict[prefix_idx])
        elif idx in test_geo3k:
            formal_test_keys.append(geo2formal_dict[prefix_idx])
        else:
            formal_train_keys.append(geo2formal_dict[prefix_idx])
    
    for prefix_idx in dataset_dict['GeoQA']:
        idx = prefix_idx.replace('GeoQA-', '')
        if idx in valid_geoqa:
            formal_val_keys.append(geo2formal_dict[prefix_idx])
        elif idx in test_geoqa:
            formal_test_keys.append(geo2formal_dict[prefix_idx])
        else:
            formal_train_keys.append(geo2formal_dict[prefix_idx])
            
    total_keys = formal_val_keys + formal_test_keys + formal_train_keys
    try:
        assert len(total_keys) == len(set(total_keys)), "List contains duplicate elements."
    except Exception as e:
        duplicates = set()
        for item in total_keys:
            # 如果元素在剩余列表中还出现过，则认为它是重复的
            if total_keys.count(item) > 1:
                duplicates.add(item)
        print(list(duplicates))
        
    formal_train_keys.sort()
    formal_val_keys.sort()
    formal_test_keys.sort()
    
    # Print the updated lengths
    print('Fgo from Train: ', len(formal_train_keys))
    print('Fgo from Valid: ', len(formal_val_keys))
    print('Fgo from Test : ', len(formal_test_keys))
    
    train_data = {k:formal_geo_data[int(k)] for k in formal_train_keys}
    valid_data = {k:formal_geo_data[int(k)] for k in formal_val_keys}
    test_data = {k:formal_geo_data[int(k)] for k in formal_test_keys}
    
    save_json(train_data, f"datasets/processed_data/fgo_train.json")
    save_json(valid_data, f"datasets/processed_data/fgo_val.json")
    save_json(test_data, f"datasets/processed_data/fgo_test.json")

    print('Split Done')
    
def stats_for_geoqa():
    geoqa_dir = "D:/Desktop/资源/几何答题/GeoQA/GeoQA3"
    train_data = read_pickle(os.path.join(geoqa_dir, 'train.pk'))
    dev_data = read_pickle(os.path.join(geoqa_dir, 'dev.pk'))
    test_data = read_pickle(os.path.join(geoqa_dir, 'test.pk'))
    print('GeoQA Origin Data')
    print('Train: ', len(train_data))
    print('Dev: ', len(dev_data))
    print('Test: ', len(test_data))
    print('')
    valid_keys = [str(data['id']) for data in dev_data]
    test_keys = [str(data['id']) for data in test_data]
    return valid_keys, test_keys
    
def stats_for_geometry3k():
    geo3k_dir = "D:/Desktop/资源/几何答题/InterGPS-main/data/geometry3k"
    train_keys = os.listdir(f"{geo3k_dir}/train")
    valid_keys = os.listdir(f"{geo3k_dir}/val")
    test_keys = os.listdir(f"{geo3k_dir}/test")
    print('Geometry3K Origin Data')
    print('Train: ', len(train_keys))
    print('Dev: ', len(valid_keys))
    print('Test: ', len(test_keys))
    print('')
    
    return valid_keys, test_keys
    
if __name__ == '__main__':
    stats_source()
    # stats_for_geoqa()
    
        