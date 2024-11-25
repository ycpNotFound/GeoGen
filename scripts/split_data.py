import json
import os
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
import itertools
import shutil

def identify_image(positions, fig_size):
    # fig y >> x or fig x >> y
    if max(fig_size) > 2.5 * min(fig_size):
        return False, f"fig size {max(fig_size)} > 2.5*{min(fig_size)}"

    # min distance < 10
    distances = []
    for pos1, pos2 in itertools.combinations(positions, 2):
        dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
        distances.append(dist) 
    if min(distances) < 10:
        return False, f"max dist {min(distances)} < 10"
    
    # there's outlier point
    avg_x = sum([p[0] for p in positions]) / len(positions)
    avg_y = sum([p[1] for p in positions]) / len(positions)
    distances_center = []
    for pos in positions:
        dist = ((pos[0] - avg_x) ** 2 + (pos[1] - avg_y) ** 2) ** 0.5
        distances_center.append(dist)
    for i, dist in enumerate(distances_center):
        other_distances = distances_center[:i] + distances_center[i+1:]
        other_avg = sum(other_distances) / len(other_distances)
        if dist > 3 * other_avg:
            return False, f"outlier point {dist} > 3*{other_avg}"
        
    return True, None
    
def main():
    # task = "ENT_1_REL_2"
    task = "pretrain"
    image_dir = f"geo_gen/geo_gen_{task}/imgs"
    final_image_dir = f"geo_gen/geo_gen_{task}/imgs_final"
    error_image_dir = f"geo_gen/geo_gen_{task}/imgs_error"
    os.makedirs(error_image_dir, exist_ok=True)
    os.makedirs(final_image_dir, exist_ok=True)
    json_path = f"geo_gen/geo_gen_{task}/annotations.json"
    json_path_final = f"geo_gen/geo_gen_{task}/annotations_final.json"
    json_data = json.load(open(json_path, 'r', encoding='utf-8'))
    
    error_cnt = 0
    success_cnt = 0
    final_json_data = {}
    for k_, v_ in tqdm(json_data.items()):
        v = deepcopy(v_)
        pos = v['positions']
        pos = [p for p in list(pos.values())]
        
        image_path = f"{image_dir}/{k_}.png"
        fig_size = Image.open(image_path).size
        
        flag, error_msg = identify_image(pos, fig_size)
        
        if flag == False:
            # copy image to error dir
            shutil.copy2(image_path, f"{error_image_dir}/{k_}.png")
            error_cnt += 1
            # print(error_msg)
            # print(image_path)
            # print('------------------------')
        else:
            
            shutil.copy2(image_path, f"{final_image_dir}/{success_cnt}.png")
            final_json_data[success_cnt] = v
            success_cnt += 1
            
    print(f"error cnt: {error_cnt}")
    print(f"success cnt: {success_cnt}")
    with open(json_path_final, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4, ensure_ascii=False)
    
if __name__ == '__main__':
    main()