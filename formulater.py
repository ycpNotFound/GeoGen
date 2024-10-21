from allocator import Allocator
from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from plotter import Plotter
import random
from utils import PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2, setup_seed
import json

from formalgeo.core import EquationKiller as EqKiller
from formalgeo.data import DatasetLoader
from formalgeo.parse import parse_theorem_seqs
from formalgeo.problem.condition import Goal
from formalgeo.solver import ForwardSearcher
from graph import ConditionGraph, ConditionNode, display_solution, draw_graph

class QuestionFormulater():
    def __init__(self, 
                 solver,
                 allocater_states, 
                 text_cdls,
                 construct_cdls):
        self.p_pos = allocater_states['p_pos']
        self.lines = allocater_states['lines']
        self.circles = allocater_states['circles']
        self.points_on_circle = allocater_states['points_on_circle']
        # self.clauses = allocater_states['clauses_base'] + allocater_states['clauses']
        self.clauses = allocater_states['clauses']
        self.text_cdls = text_cdls
        self.construct_cdls = construct_cdls

        self.solver: ForwardSearcher = solver
        
    def solve(self):
        pass
        
        
        
if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(10):
        # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
        clauses_base = random.choices(PREDICATES_ENT, k=1)
        clauses_rel = random.choices(PREDICATES_REL, k=2)
        clauses_base = [
            "RightTrapezoid",
        ]
        clauses_rel = [
            'IsBisectorOfAngle', 
            # 'IsMidsegmentOfTriangle',
            # 'IsAltitudeOfQuadrilateral',
            # 'IsIncenterOfTriangle',
            # "IsAltitudeOfTriangle",
            # "IsCircumcenterOfQuadrilateral",
            # "IsMidpointOfArc"
            ]
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_more_lines=0
        )
        states = cg.states
        
        # states = {'points': ['a', 'b', 'c', 'd', 'e'], 'lines': [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd'), ('d', 'e'), ('a', 'c')], 'circles': [], 'polygons': [('a', 'b', 'c', 'd'), ('a', 'b', 'c'), ('a', 'c', 'd')], 'constraints': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)', 'Equal(MeasureOfAngle(ade),MeasureOfAngle(edc))'], 'constraints_base': ['ParallelBetweenLine(ad,bc)', 'Equal(MeasureOfAngle(dab),90)', 'Equal(MeasureOfAngle(abc),90)'], 'points_on_circle': {}}
        # c_cdls = ['Shape(ab,bc,cd,da)', 'Shape(de)']
        # t_cdls = ['RightTrapezoid(abcd)', 'IsBisectorOfAngle(de,adc)']  
        
        print('---------- Allocator Inputs ----------')
        print(states)
        print('c_cdls: ', c_cdls)
        print('t_cdls: ', t_cdls)

        allocator = Allocator(states, c_cdls, t_cdls)
        allocator.allocate()
        print('---------- Formulated CDLs ----------')
        print('Text CDLs: ')
        for t_cdl in allocator.formulated_cdls['text_cdls']:
            print('\t', t_cdl)
        print('Construct CDLs: ')
        for c_cdl in allocator.formulated_cdls['construct_cdls']:
            print('\t', c_cdl)
            
        plotter = Plotter(allocator.states, 
                          allocator.formulated_cdls['text_cdls'],
                          allocator.formulated_cdls['construct_cdls'])
        plotter.plot()
        plotter.save_fig('test', 'imgs_test')
        
        t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
        solver = ForwardSearcher(
            dl.predicate_GDL, dl.theorem_GDL, 
            strategy="dfs", max_depth=15, 
            beam_size=10, t_info=t_info,
            debug=True
        )
        formulater = QuestionFormulater(
            solver, 
            allocator.states, 
            allocator.formulated_cdls['text_cdls'],
            allocator.formulated_cdls['construct_cdls']
        )
        t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))

        print('==============================================')