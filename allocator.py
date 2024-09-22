import random
import re

import sympy
from sympy import Eq, Expr, Float, cos, pi, simplify, solve, symbols

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils import (get_content, get_points, get_predicate_name, get_symbol,
                   max_letter_index, parse_clause, replace_points, setup_seed, find_target_for_construct)

# ClauseGenerator 
# Allocator
# Plotter
# Solver
# ConditionGraph

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
        
        
        self.clauses = self.construct_cdls + self.constraints
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
        
    def find_mini_clauses_subset(self):
        # find min clauses subset to construct each point 
        targets_in_clauses = [find_target_for_construct(c) for c in self.clauses] 
        clause_subset = {p: [] for p in self.points}
        for targets, clause in zip(targets_in_clauses, self.clauses):
            for p in targets:
                clause_subset[p].append(clause)
                
        return clause_subset
    
    def merge_cocircular_clauses(self):
        # merge all cocircular clauses into 2 clauses:
        # 1. to construct circle centre, e.g. abc -> o
        # 2. to construct other points greater than circle, e.g. o -> defg
        cocircular_clauses = [c for c in self.clauses if 'Cocircular' in c]
        other_clauses = [c for c in self.clauses if 'Cocircular' not in c]
        points_on_circle = {}
        circles = set()
        for c in cocircular_clauses:
            _, items = parse_clause(c)
            circle, p_on_circle = items
            if circle in points_on_circle:
                points_on_circle[circle] += list(p_on_circle)
            else:
                points_on_circle[circle] = list(p_on_circle)
            circles.add(circle)

        clause_merged_list = []
        for circle in list(circles):
            p_on_circle = list(set(points_on_circle[circle]))
            p_greater_than_circle = sorted([p for p in p_on_circle if p > circle])
            p_less_than_circle = sorted([p for p in p_on_circle if p < circle])
            
            # clause to construct circle centre
            # p less than circle -> construc circle
            if len(p_less_than_circle) > 0:
                clause_merged_1 = f"Cocircular({circle},{''.join(p_less_than_circle)})"
                clause_merged_list.append(clause_merged_1)
            
            # clause to construct other points on circle
            # circle, first p on circle -> construct other p greater than circle
            if len(p_greater_than_circle) > 0:
                clause_merged_2 = f"Cocircular({circle},{''.join(p_less_than_circle[:1] + p_greater_than_circle)})"
                clause_merged_list.append(clause_merged_2)

        
        return clause_merged_list + other_clauses
        
            
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
        elif name == 'ParallelBetweenLine':
            self.allocate_parallel(clause)
        elif name in ['CongruentBetweenQuadrilateral', 
                      'CongruentBetweenTriangle']:
            self.allocate_congruent(clause)
        elif name in ['SimilarBetweenQuadrilateral', 
                      'SimilarBetweenTriangle']:
            self.allocate_similar(clause)

        self.update_position(clause)
        
    def allocate_equal(self, clause):
        predicate, items = parse_clause(clause)
        if predicate == 'MeasureOfAngle':
            self.allocate_equal_angle(items[0], items[1])
        elif predicate == 'LengthOfLine':
            self.allocate_equal_line(items[0], items[1])
        elif predicate == 'LengthOfArc':
            self.allocate_equal_arc(items[0], items[1])
        else:
            print('Error: ', predicate, items)
            raise KeyError(predicate)

    def allocate_parallel(self, clause):
        predicate, items = parse_clause(clause)
        line_1, line_2 = items[0], items[1]
        points = list(set(line_1 + line_2))
        self.define_points(points)
        x1, y1 = self.p_pos[line_1[0]]
        x2, y2 = self.p_pos[line_1[1]]
        x3, y3 = self.p_pos[line_2[0]]
        x4, y4 = self.p_pos[line_2[1]]
        
        k1 = (y2 - y1) / (x2 - x1)
        k2 = (y4 - y3) / (x4 - x3)
        eq = Eq(k1, k2)
        char, target, expand_eq = self.find_target(points)
        solution = solve((eq, ) + expand_eq, target)
        self.update_symbol(solution, char)
        
        return 
    
    def allocate_collinear(self, clause):
        predicate, items = parse_clause(clause)
        points = items[0]
        self.define_points(points)
        sorted_p = sorted(points)
        # two points ensure one line
        # sort and choose the first 2 points
        x1, y1 = self.p_pos[sorted_p[0]]
        x2, y2 = self.p_pos[sorted_p[1]]

        for other_p in sorted_p[2:]:
            xi, yi = self.p_pos[other_p]
            char, target, expand_eq = self.find_target([other_p])
            eq = Eq((yi - y1) / (xi - x1), (y2 - y1) / (x2 - x1))
            if target is None:
                return False

            solution = solve((eq, ) + expand_eq, target)
            self.update_symbol(solution, other_p)
        
        return
    
    def allocate_cocircular(self, clause):
        predicate, items = parse_clause(clause)
        circle, points = items[0], items[1]
        self.define_points([circle] + list(points))
        
        if circle > max(points[:3]):
            # define new circle center
            char, target, expand_eq = self.find_target([circle])
            if target is None:
                return
            circle_eqs = self.get_circum_circle_eqs(target, points[:3])
            solution = solve(circle_eqs + expand_eq, target)
            self.update_symbol(solution, circle)
            
            # define other points (if) on circle
            r_len_2 = self.get_line_length_2([circle, points[0]])
            for p in points[3:]:
                r_len_i_2 = self.get_line_length_2([circle, p])
                eq = Eq(r_len_2, r_len_i_2)
                char, target, expand_eq = self.find_target([p])
                solution = solve((eq, ) + expand_eq, target)
                self.update_symbol(solution, p)
        else: 
            # define other points (if) on circle
            sorted_points = sorted(points)
            r_len = self.get_line_length_2([circle, sorted_points[0]])
            for p in sorted_points[1:]:
                r_len_i = self.get_line_length_2([circle, p])
                eq = Eq(r_len_i, r_len)
                char, target, expand_eq = self.find_target([p])
                if target is None:
                    continue
                solution = solve((eq, ) + expand_eq, target)
                self.update_symbol(solution, p)
        
        return
    
    def allocate_congruent(self, pclause):
        predicate, items = parse_clause(clause)
        points_1, points_2 = items
        self.define_points(list(points_1 + points_2))
    
    def allocate_similar(self, clause):
        predicate, items = parse_clause(clause)
        points_1, points_2 = items
        self.define_points(list(points_1 + points_2))
        

    def allocate_equal_angle(self, angle_1, angle_2):
        if not angle_1.isdigit(): 
            self.define_points(angle_1)
        if not angle_2.isdigit(): # angle abc = angle def
            self.define_points(angle_2)
        # get expression of cos
        cos_1 = self.get_cos(angle_1)
        cos_2 = self.get_cos(angle_2)
        if isinstance(cos_1, float) and isinstance(cos_2, float):
            if abs(cos_1 - cos_2) < 1e-8:
                return
            else:
                raise ValueError(cos_1 - cos_2)
        angle_eq = Eq(cos_1, cos_2)
        # find the max index point as target
        if angle_2.isdigit():
            char, target, expand_eq = self.find_target(angle_1)
        else:
            char, target, expand_eq = self.find_target(angle_1 + angle_2)
        # solve the equation
        if target is None:
            return False
        
        solution = solve((angle_eq, ) + expand_eq, target)
        # update position of target point
        self.update_symbol(solution, char) 
        return 
        
    
    def allocate_equal_line(self, line_1, line_2):
        self.define_points(line_1)
        self.define_points(line_2)
        len_1 = self.get_line_length(line_1)
        len_2 = self.get_line_length(line_2)
        if isinstance(len_1, float) and isinstance(len_2, float):
            if abs(len_1 - len_2) < 1e-8:
                return
            else:
                raise ValueError(len_1 - len_2)
            
        eq = Eq(len_1, len_2)
        char, target, expand_eq = self.find_target(line_1 + line_2)
        if target is None:
            return False

        solution = solve((eq, ) + expand_eq, target)
        self.update_symbol(solution, char) 
        return 
    
    def allocate_equal_arc(self, arc_1, arc_2):
        if not arc_1.isdigit(): 
            self.define_points(arc_1)
        if not arc_2.isdigit(): # angle abc = angle def
            self.define_points(arc_2)
            
        circle, p1, p2 = arc_1
        circle, p3, p4 = arc_2
        dot_1 = self.get_dot_product([circle, p1], [circle, p2])
        dot_2 = self.get_dot_product([circle, p3], [circle, p4])
        
        if isinstance(dot_1, float) and isinstance(dot_2, float):
            if abs(dot_1 - dot_2) < 1e-8:
                return
            else:
                raise ValueError(dot_1 - dot_2)
        eq = Eq(dot_1, dot_2)
        # find the max index point as target
        if arc_2.isdigit():
            char, target, expand_eq = self.find_target(arc_1)
        else:
            char, target, expand_eq = self.find_target(arc_1 + arc_2)
        # solve the equation
        if target is None:
            return False
        
        solution = solve((eq, ) + expand_eq, target)
        # update position of target point
        self.update_symbol(solution, char) 
        return 
        
            
    
    def get_cos(self, angle):
        # angle ABC
        if angle.isdigit():
            return cos(float(angle) * pi / 180)
        p1, p2, p3 = angle
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        xc, yc = self.p_pos[p3]
        BA = (xa - xb, ya - yb)
        BC = (xc - xb, yc - yb)
        dot_product = BA[0]*BC[0] + BA[1]*BC[1]
        len_BA = (BA[0]**2 + BA[1]**2)**0.5
        len_BC = (BC[0]**2 + BC[1]**2)**0.5
        return dot_product / (len_BA * len_BC)
    
    def get_dot_product(self, line_1, line_2):
        # AB * CD 
        p1, p2 = line_1
        p3, p4 = line_2
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        xc, yc = self.p_pos[p3]
        xd, yd = self.p_pos[p4]
        AB = (xb - xa, yb - ya)
        CD = (xd - xc, yd - yc)
        return AB[0]*CD[0] + AB[1]*CD[1]
    
    def get_line_length(self, line):
        p1, p2 = line
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        len_BA = ((xa - xb)**2 + (ya - yb)**2)**0.5
        return len_BA
    
    def get_line_length_2(self, line):
        p1, p2 = line
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        len_BA_2 = ((xa - xb)**2 + (ya - yb)**2)
        return len_BA_2
    
    def get_radius_length(self):
        pass
    
    def get_circum_circle_eqs(self, target, points):
        
        x, y = target
        points = sorted(points)

        if len(points) >= 3:
            # get centre of circum_circle of ABC
            a, b, c = points[:3]
            xa, ya = self.p_pos[a]
            xb, yb = self.p_pos[b]
            xc, yc = self.p_pos[c]
            # m, n is midpoint of AB, BC
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            xn, yn = (xb + xc) / 2, (yb + yc) / 2
            
            k_ab = (ya - yb) / (xa - xb) 
            k_bc = (yb - yc) / (xb - xc)
            k_perp_ab = -1 / k_ab 
            k_perp_bc = -1 / k_bc 
            
            eq1 = Eq(y - ym, k_perp_ab * (x - xm))
            eq2 = Eq(y - yn, k_perp_bc * (x - xn))
            return (eq1, eq2)
        
        elif len(points) == 2:
            a, b = points
            xa, ya = self.p_pos[a]
            xb, yb = self.p_pos[b]
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            k_ab = (ya - yb) / (xa - xb)
            k_perp_ab = -1 / k_ab 
            eq1 = Eq(y - ym, k_perp_ab * (x - xm))
            return (eq1, )
        
        else:
            raise IndexError(points)
            
    
    def define_points(self, points):
        # only define points when used in clause
        for p in points:
            if self.p_pos[p] == None:
                self.p_pos[p] = get_symbol(p)
                
        return [self.p_pos[p] for p in points]

    def find_target(self, letters):
        # find target symbol, may return related equation
        for c in sorted(letters, reverse=True):
            x, y = self.p_pos[c]
            x_sym, y_sym = get_symbol(c)
            x_solved, y_solved = False, False
            if x != x_sym:
                x_solved = True
            if y != y_sym:
                y_solved = True
            if x_solved and y_solved:
                # find next not solved symbol
                continue
            elif x_solved and not y_solved:
                return c, tuple([x_sym, y_sym]), (Eq(x_sym, x), )
            elif not x_solved and y_solved:
                return c, tuple([x_sym, y_sym]), (Eq(y_sym, y), )
            elif not x_solved and not y_solved:
                return c, tuple([x_sym, y_sym]), ()
        return None, None, ()
    
    def update_symbol(self, solution, char, ignore_value=None):
        # update p_pos with new solved symbols
        if len(solution) == 0:
            return False
        if type(solution) == dict:
            syms = get_symbol(char)
            x = solution.get(syms[0], self.p_pos[char][0])
            y = solution.get(syms[1], self.p_pos[char][1])
        elif type(solution) == list:
            # have multiple solutions, choose reasonable (don't have big numbers)
            x, y = random.choice(solution)
        elif type(solution) == tuple:
            x, y = solution
        else:
            print('Error: ', type(solution))
            raise TypeError(solution)
            
        if isinstance(x, Float):
            x = float(x)
        if isinstance(y, Float):
            y = float(y)
            
        self.p_pos[char] = [x, y]
            
    def update_position(self, clause):
        # update point positions
        # check for subset first
        for p, subset in self.clause_subset.items():
            if clause in subset:
                # this can modify self.clause_subset dict
                subset.remove(clause)
        # only random allocate position when:
        # 1. have visited all clauses in subset
        # 2. have defined symbols for this point
        # 3. p_pos[p] is not number, and have unsolved symbols
        for p in self.points:
            if len(self.clause_subset[p]) == 0:
                if self.p_pos[p] is not None:
                    x, y = self.p_pos[p]
                    if isinstance(x, Expr) or isinstance(y, Expr):
                        position = self.random_allocate_position(p)
                        self.update_values(p, position)
        
    def random_allocate_position(self, point):
        x = random.uniform(5, 10)
        y = random.uniform(5, 10)
        return [x, y]
            
    def update_values(self, point, position):
        # point: char; position: float value
        x_value, y_value = position
        x_sym, y_sym = get_symbol(point)
        # export x_sym = x_value, y_sym = y_value in all expressions
        for p in self.points:
            if self.p_pos[p] is None:
                continue
            x_p, y_p = self.p_pos[p]
            if isinstance(x_p, Expr):
                if x_sym in x_p.free_symbols:
                    x_p = x_p.subs(x_sym, x_value)
                if y_sym in x_p.free_symbols:
                    x_p = x_p.subs(y_sym, y_value)
                x_p = simplify(x_p)
            if isinstance(y_p, Expr):
                if x_sym in y_p.free_symbols:
                    y_p = y_p.subs(x_sym, x_value)
                if y_sym in y_p.free_symbols:
                    y_p = y_p.subs(y_sym, y_value)
                y_p = simplify(y_p)
            
            if isinstance(x_p, Float):
                x_p = float(x_p)
            if isinstance(y_p, Float):
                y_p = float(y_p)
            self.p_pos[p] = [x_p, y_p]

            
    
            

if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(5):
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        constr_cdls, text_cdls = cg.generate_clauses_from_predicates(1, 2, 1, 2)
        allocator = Allocator(cg.states, constr_cdls, text_cdls, dl.predicate_GDL)
        
        print(f"{'-'*10} Clause {'-'*10}")
        for clause in allocator.clauses:
            print(clause)
            
        allocator.allocate()
        print(f"{'-'*10} Location {'-'*10}")
        for p, pos in allocator.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
    
        print('==================================')

        