import json
import random
import sys
from pathlib import Path

# 获取当前文件的绝对路径
current_path = Path(__file__).resolve()
# 获取当前文件所在的目录
parent_path = current_path.parent
# 获取当前目录的父目录
grandparent_path = parent_path.parent

# 将父目录添加到 sys.path 中
sys.path.append(str(grandparent_path))

from utils import setup_seed
from plotter import Plotter
from generator import ClauseGenerator
from allocator import Allocator
 

def re_generate(
    key,
    pred_base,
    pred_rel,
    n_more_lines,
    color_config,
    fig_dir
):
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    cg = ClauseGenerator(predicate_GDL, theorem_GDL)
    cg.empty_states()
    c_cdls, t_cdls = cg.generate_clauses_from_predicates(
        pred_base, 
        pred_rel, 
        n_more_lines=n_more_lines
    )
    states = cg.states

    print('---------- Allocator Inputs ----------')
    print(states)
    print('c_cdls: ', c_cdls)
    print('t_cdls: ', t_cdls)

    allocator = Allocator(states, c_cdls, t_cdls, allocate_value=True)
    print('---------- Formulated CDLs ----------')
    
        
    allocator.allocate()
    print('Text CDLs: ')
    for t_cdl in allocator.formulated_cdls['text_cdls']:
        print('\t', t_cdl)
    print('Construct CDLs: ')
    for c_cdl in allocator.formulated_cdls['construct_cdls']:
        print('\t', c_cdl)
        
    plotter = Plotter(allocator.states, 
                        allocator.formulated_cdls['text_cdls'],
                        allocator.formulated_cdls['construct_cdls'],
                        allocator.formulated_cdls['image_cdls'],
                        color_config=color_config,
                        replace_characters=True)

    
    plotter.plot()
    fig_name = f"{key}.png"
    plotter.save_fig(fig_dir=fig_dir, fig_name=fig_name)
    data_info = {
        "construction_cdl": allocator.formulated_cdls['construct_cdls'],
        "text_cdl": allocator.formulated_cdls['text_cdls'], 
        "image_cdl": plotter.image_cdls,
        "caption_str": plotter.caption_str
    }
    return data_info
    

def re_generate_main():
    fail_data = json.load(open('geo_gen/geo_gen_pretrain/failure_cases.json', 'r', encoding='utf-8'))
    annot_data = json.load(open('geo_gen/geo_gen_pretrain/annotations.json', 'r', encoding='utf-8'))
    fig_dir = 'geo_gen/geo_gen_pretrain/imgs'
    for key, value in fail_data.items():
        info = re_generate(
            key,
            value['pred_base'],
            value['pred_rel'],
            value['n_more_lines'],
            value['color_config'],
            fig_dir=fig_dir
        )
        print(f"geo_gen/geo_gen_pretrain/imgs/{key}.png")
        annot_data[key] = info
        
    with open('geo_gen/geo_gen_pretrain/annotations.json', 'w', encoding='utf-8') as f:
        json.dump(annot_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    setup_seed(114)
    re_generate_main()
    