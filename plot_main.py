
import random

import itertools
from allocator import Allocator
from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from plotter import Plotter
from utils import PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2, setup_seed

if __name__ == '__main__':
    setup_seed(124)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
    predicate_base_combs = itertools.permutations(PREDICATES_ENT, 1)
    for predicate_base in predicate_base_combs:
        predicate_rel_combs = itertools.permutations(PREDICATES_REL, 2)
        for predicate_rel in predicate_rel_combs:
            print('---------- Chosen Predicates ----------')
            print('clauses_base: ', predicate_base)
            print('clauses_rel: ', predicate_rel)
            
            cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
            cg.empty_states()
            c_cdls, t_cdls = cg.generate_clauses_from_predicates(
                predicate_base, 
                predicate_rel, 
                n_more_lines=1
            )
            states = cg.states
        
            print('---------- Allocator Inputs ----------')
            print(states)
            print('c_cdls: ', c_cdls)
            print('t_cdls: ', t_cdls)

            allocator = Allocator(states, c_cdls, t_cdls, dl.predicate_GDL)
            print('---------- Formulated CDLs ----------')
            
                
            allocator.allocate()
            print('Text CDLs: ')
            for t_cdl in allocator.formulated_cdls['text_cdls']:
                print('\t', t_cdl)
            print('Construct CDLs: ')
            for c_cdl in allocator.formulated_cdls['construct_cdls']:
                print('\t', c_cdl)
                
            plotter = Plotter(allocator.states)
            plotter.plot()
            plotter.save_fig('test', 'imgs_test')
            
            print('==============================================')