import random
import re

import sympy
from sympy import Eq, Expr, Float, cos, pi, simplify, solve, symbols

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils import (PREDICATES_ENT, PREDICATES_REL, find_target_for_construct,
                   get_content, get_points, get_predicate_name, get_symbol,
                   max_letter_index, parse_clause, replace_points, setup_seed)

# ClauseGenerator 
# Allocator
# Plotter
# Solver

END_NAMES = [
    'Equal', 'ParallelBetweenLine', 'IsCircumcenterOfTriangle',
    'IsCentroidOfTriangle', 
    'CongruentBetweenTriangle', 'MirrorCongruentBetweenTriangle',
    'SimilarBetweenTriangle', 'MirrorSimilarBetweenTriangle', 
    'IsCircumcenterOfQuadrilateral', 'CongruentBetweenQuadrilateral',
    'MirrorCongruentBetweenQuadrilateral', 'SimilarBetweenQuadrilateral', 
    'CongruentBetweenArc', 'IsDiameterOfCircle', 'IsTangentOfCircle',
]
IGNORE_NAMES = [
    'Shape',
    'IsCircumcenterOfQuadrilateral',
    'IsCircumcenterOfTriangle',
]
    
def get_max_letter(s):
    return sorted(s)[-1]
    

class Allocator():
    def __init__(self, geo_states, construct_cdls, text_cdls, predicate_GDL):
        self.points = geo_states['points']                     
        self.lines = geo_states['lines']
        self.circles = geo_states['circles']
        self.constraints = geo_states['constraints']
        self.construct_cdls = construct_cdls
        self.text_cdls = text_cdls
        self.predicate_GDL = predicate_GDL
        self.predicate_ent_names = {s.split('(')[0]: s for s in predicate_GDL['Entity']}
        self.predicate_rel_names = {s.split('(')[0]: s for s in predicate_GDL['Relation']}
        # ee_check: geo entity (line segment / circle / arc)
        # extend: numerical (length / angle measure)
        
        
        self.clauses = self.construct_cdls + self.text_cdls
        # merge all cocircular clauses
        self.clauses = self.merge_cocircular_clauses()
        # sort by idx of points
        self.clauses = sorted(self.clauses, key=max_letter_index)
        # filter by IGNORE_NAMES
        self.clauses = [c for c in self.clauses if c.split('(')[0] not in IGNORE_NAMES]

        # store the location of points
        self.p_pos = {k: None for k in self.points}
        self.p_pos['a'] = [0, 0]

        # find min clauses subset needed to construct each point 
        # key: point | value: subset of clauses
        # if all clauses are visited in one subset
        # then random allocate position for this point
        self.clause_subset = self.find_mini_clauses_subset()
        
    def allocate(self):
        for clause in self.clauses:
            self.allocate_for_one_clause(clause)
            
        # if there're still points not solved:
        for p in self.points:
            x, y = self.p_pos[p]
            if isinstance(x, Expr) or isinstance(y, Expr):
                position = self.random_allocate_position(p)
                self.update_values(p, position)
            
    def allocate_for_one_clause(self, clause):
        name = get_predicate_name(clause)

        if name == 'Collinear':
            self.allocate_collinear(clause)
        elif name == 'Cocircular':
            self.allocate_cocircular(clause)
        elif name == 'Equal':
            self.allocate_equal(clause)
        
        elif name in ['CongruentBetweenQuadrilateral', 
                      'CongruentBetweenTriangle',
                      'CongruentBetweenArc']:
            self.allocate_congruent(clause)
        elif name in ['SimilarBetweenQuadrilateral', 
                      'SimilarBetweenTriangle']:
            self.allocate_similar(clause)
        elif name == 'RightTriangle':
            
        elif name == 'IsoscelesTriangle':
            self.allocate_isosceles_triangle(clause)
        
            
        elif name == 'IsoscelesRightTriangle':
            
        elif name == 'EquilateralTriangle':
            
        elif name in ['Kite', 'Parallelogram', 'Rhombus', 'Rectangle', 
                      'Square', 'Trapezoid', 'IsoscelesTrapezoid', 'RightTrapezoid']:
        
        elif name in ['IsMidpointOfLine', 'IsMidpointOfArc'] :
        
        elif name == 'ParallelBetweenLine':
            self.allocate_parallel(clause)
        
        elif name == 'PerpendicularBetweenLine':
        
        elif name == 'PerpendicularBetweenLine':
            
        elif name == 'IsPerpendicularBisectorOfLine':
            
        elif name == 'IsBisectorOfAngle'

        self.update_position(clause)
        
    def allocate_collinear(self, clause):
        pass
    
    def allocate_cocircular(self, clause):
        pass
    
    def allocate_parallel(self, clause):
        pass
    
    def allocate_congruent(self, clause):
        pass
    
    def allocate_similar(self, clause):
        pass
    
    def allocate_isosceles_triangle(self, clause):
        pass
    
    
            
if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(5):
        clauses_entity = random.choices(PREDICATES_ENT, k=1)
        clauses_rel = random.choices(PREDICATES_REL, k=2)
        # clauses_entity = ['Rhombus']
        # clauses_rel = ['IsMidsegmentOfTriangle', 'CongruentBetweenArc']
        
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_entity, 
            clauses_rel, 
            n_new_lines=2
        )
        print('---------- Chosen Predicates ----------')
        print('clauses_entity: ', clauses_entity)
        print('clauses_rel: ', clauses_rel)
        
        print('---------- Construct CDLs ----------')
        for c_cdl in c_cdls:
            print(c_cdl)
        
        print('---------- Text CDLs ----------')
        for t_cdl in t_cdls:
            print(t_cdl)
        
        print('---------- States ----------')
        cg.print_states()
        
        allocator = Allocator(cg.states, c_cdls, t_cdls, dl.predicate_GDL)
        allocator.allocate()
        
        print("---------- Location ----------")
        for p, pos in allocator.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
    
        print('==============================================')

        