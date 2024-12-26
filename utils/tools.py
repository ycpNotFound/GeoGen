import random

import numpy as np


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