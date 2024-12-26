import random

import numpy as np
import itertools

def setup_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    

def remove_duplicates(lst):
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result

def append_lst(lst, items: list):
    for item in items:
        if item not in lst:
            lst.append(item)
            
    return lst

def identify_image(positions, fig_size):
    # fig y >> x or fig x >> y
    if isinstance(positions, dict):
        positions = [p for p in positions.values()]
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