import json
import os

def test_1():
    theorem_freq = json.load(open('json/theorem_freq.json', encoding='utf-8'))
    theorem_zero = [k for k in theorem_freq if theorem_freq[k] == 0]

    theorem_gdl = json.load(open('json/theorem_GDL.json', encoding='utf-8'))
    theorem_gdl_new = {}
    for k, v in theorem_gdl.items():
        name = k.split('(')[0]
        if name not in theorem_zero:
            theorem_gdl_new[k] = v
    with open('json/theorem_GDL_new.json', 'w', encoding='utf-8') as f:
        json.dump(theorem_gdl_new, f, indent=2, ensure_ascii=False)

def test_2():
    from utils import parse_clause
    name, items = parse_clause('Equation(ll_ac-sqrt(2)*ll_bd)')
    print(name)
    print(items)
    
if __name__ == '__main__':
    test_2()