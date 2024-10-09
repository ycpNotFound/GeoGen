import random
import re
import math
import sympy
from sympy import Eq, Expr, Symbol, Float, cos, pi, simplify, solve, symbols, I, im, nsimplify

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils import (PREDICATES_ENT, PREDICATES_REL, find_target_for_construct,
                   get_content, get_points, get_predicate_name, get_symbol,
                   max_letter_index, parse_clause, replace_points, setup_seed, simplify_and_trim)

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
        assert len(self.circles) in [0, 1]
        self.constraints = [c for c in geo_states['constraints'] 
                            if c not in geo_states['constraints_base']]
        self.constraints_base = geo_states['constraints_base']
        self.points_on_circle = geo_states['points_on_circle']
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
        
    @property
    def states(self):
        return {
            "p_pos": self.p_pos,
            "points": self.points,
            "lines": self.lines,
            "circles": self.circles,
            "points_on_circle": self.points_on_circle,
        }
        
    @property
    def formulated_cdls(self):
        construct_cdls = []
        cocircular_cdls = []
        for const_cdl in self.construct_cdls:
            if 'Shape' in const_cdl or 'Collinear' in const_cdl:
                construct_cdls.append(const_cdl)
            if 'Cocircular' in const_cdl:
                cocircular_cdls.append(const_cdl)

        if len(cocircular_cdls) > 0:
            points = set()
            for cdl in cocircular_cdls:
                _, items = parse_clause(cdl)
                circle, ps_on_circle = items
                points.update(ps_on_circle)
            cocircular_cdl_total = f"Cocircular({circle},{''.join(sorted(list(points)))})"
            construct_cdls.append(cocircular_cdl_total)
            
        return {
            "text_cdls": self.text_cdls,
            "construct_cdls": construct_cdls
        }
        
    def empty_states(self):
        self.p_pos = {k: None for k in self.points}
        self.p_pos['a'] = [0, 0]
        self.clause_subset = self.find_mini_clauses_subset()
        
    @staticmethod
    def distance_2(point1, point2):
        return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def find_mini_clauses_subset(self):
        # find min clauses subset to construct each point 
        targets_in_clauses = [find_target_for_construct(c) for c in self.clauses] 
        clause_subset = {p: [] for p in self.points}
        for targets, clause in zip(targets_in_clauses, self.clauses):
            for p in targets:
                if clause not in clause_subset[p]:
                    clause_subset[p].append(clause)
                
        return clause_subset
    
    def merge_cocircular_clauses(self):
        # merge all cocircular clauses into 2 clauses:
        # e.g. Cocircular(o, abcde)
        # 1. to construct circle centre, e.g. abc -> o
        # 2. to construct other points greater than abc, e.g. o -> de
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
            p_smaller = sorted(p_on_circle)[:3]
            p_greater = sorted(p_on_circle)[3:]
            
            # clause to construct circle centre
            # p abc -> construc circle
            if len(p_smaller) > 0:
                clause_merged_1 = f"Cocircular({circle},{''.join(p_smaller)})"
                clause_merged_list.append(clause_merged_1)
            
            # clause to construct other points on circle
            # circle, first p on circle -> construct other p de
            if len(p_greater) > 0:
                clause_merged_2 = f"Cocircular({circle},{''.join(p_smaller[:1] + p_greater)})"
                clause_merged_list.append(clause_merged_2)

        
        return clause_merged_list + other_clauses
        
            
    def allocate(self):
        for i in range(5):
            self.allocate_for_base(self.text_cdls[0])
            
            for clause in self.clauses:
                self.allocate_for_relation(clause)
                
            # if there're still points not solved:
            for p in self.points:
                if self.p_pos[p] is None:
                    position = self.random_allocate_position()
                    self.update_values(p, position)
                else:
                    x, y = self.p_pos[p]
                    if isinstance(x, Expr) or isinstance(y, Expr):
                        position = self.random_allocate_position()
                        self.update_values(p, position)
                    
            # check if there's complex solution: a + bI
            success = True
            for p, pos in self.p_pos.items():
                x, y = pos
                if im(x).simplify() != 0 or im(y).simplify() != 0:
                    success = False
                    
            if success:
                return
            else:
                print('Get complex solutions. ')
                self.empty_states()
            
        print('Fail to allocate positions, always get complex solutions. ')
            
    def allocate_for_base(self, clause):
        name = get_predicate_name(clause)
        if name == 'Circle':
            self.allocate_circle(clause)
        if name == 'Triangle':
            self.allocate_triangle(clause)
        if name == 'RightTriangle':
            self.allocate_right_triangle(clause)
        elif name == 'IsoscelesTriangle':
            self.allocate_isosceles_triangle(clause)
        elif name == 'IsoscelesRightTriangle':
            self.allocate_isosceles_right_triangle(clause)
        elif name == 'EquilateralTriangle':
            self.allocate_equilateral_triangle(clause)
        elif name == 'Kite':
            self.allocate_kite(clause)
        elif name == 'Parallelogram':
            self.allocate_parallelogram(clause)
        elif name == 'Rhombus':
            self.allocate_rhombus(clause)
        elif name == 'Rectangle':
            self.allocate_rectangle(clause)
        elif name == 'Square':
            self.allocate_square(clause)
        elif name == 'Trapezoid':
            self.allocate_trapezoid(clause)
        elif name == 'IsoscelesTrapezoid':
            self.allocate_isosceles_trapezoid(clause)
        elif name == 'RightTrapezoid':
            self.allocate_right_trapezoid(clause)
        elif name in ['SimilarBetweenTriangle',
                      'CongruentBetweenTriangle',
                      'SimilarBetweenQuadrilateral',
                      'CongruentBetweenQuadrilateral']:
            self.allocate_congruent_similar_base(clause)
    
    def allocate_circle(self, clause):
        predicate, items = parse_clause(clause)
        point = items[0][0]
        self.define_points(point)
        if point == 'a':
            return
        else:
            self.p_pos[point] = self.random_allocate_position()
    
    def allocate_triangle(self, clause):
        # 将一点旋转随机角度，缩放随机长度得到C
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        if self.p_pos[A] is None or type(self.p_pos[A][0]) in [Expr, Symbol]:
            xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
            self.p_pos[A] = [xa, ya]
            
        if self.p_pos[B] is None or type(self.p_pos[B][0]) in [Expr, Symbol]:
            xb, yb = self.random_allocate_position()
            self.p_pos[B] = [xb, yb]
        
        xa, ya = self.p_pos[A]
        xb, yb = self.p_pos[B]
        if self.p_pos[C] is None or type(self.p_pos[C][0]) in [Expr, Symbol]:
            # 使用cv2绘图，逆时针布局需要顺时针旋转
            top_angle = - math.radians(random.uniform(45, 135)) 
            ratio = random.uniform(0.5, 1.5)
            cos_val = math.cos(top_angle)
            sin_val = math.sin(top_angle)
            xc = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
            yc = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio
            self.p_pos[C] = [xc, yc]
    
    def allocate_quad(self, clause):
        _, items = parse_clause(clause)
        A, B, C, D = items[0]
        
        if self.p_pos[A] is None or type(self.p_pos[A][0]) in [Expr, Symbol]:
            xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
            self.p_pos[A] = [xa, ya]
            
        if self.p_pos[B] is None or type(self.p_pos[B][0]) in [Expr, Symbol]:
            xb, yb = self.random_allocate_position()
            self.p_pos[B] = [xb, yb]
            
        xa, ya = self.p_pos[A]
        xb, yb = self.p_pos[B]
        
        if self.p_pos[C] is None or type(self.p_pos[C][0]) in [Expr, Symbol]:
            angle_BAC = - math.radians(random.uniform(30, 60)) 
            ratio_AC = random.uniform(1.1, 1.4)
            cos_val = math.cos(angle_BAC)
            sin_val = math.sin(angle_BAC)
            xc = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio_AC
            yc = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio_AC
            self.p_pos[C] = [xc, yc]
            
        if self.p_pos[D] is None or type(self.p_pos[D][0]) in [Expr, Symbol]:
            angle_BAD = - math.radians(random.uniform(100, 150)) 
            cos_val = math.cos(angle_BAD)
            sin_val = math.sin(angle_BAD)
            ratio_AD = random.uniform(0.6, 1.3)
            xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio_AD
            yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio_AD
            xd, yd = xd * ratio_AD, yd * ratio_AD
            self.p_pos[D] = [xd, yd]

    def allocate_right_triangle(self, clause):
        # 确定垂直单位向量，随机长度
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        BA = [xb - xa, yb - ya]
        BA_length = self.distance([xa, yb], [xb, yb])
        BC_length = random.uniform(0.75, 1.5) * BA_length
        unit_dx = (xb - xa) / BA_length
        unit_dy = (yb - ya) / BA_length

        perp_dx, perp_dy = unit_dy, -unit_dx
        xc, yc = xb + BC_length * perp_dx, yb + BC_length * perp_dy
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        
    def allocate_isosceles_triangle(self, clause):
        # 将一点旋转随机角度, A为顶角
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        
        top_angle = -math.radians(random.uniform(45, 135)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        xc = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yc = ya + (xb - xa) * sin_val + (yb - ya) * cos_val
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
            
    def allocate_isosceles_right_triangle(self, clause):
        # 确定垂直单位向量，固定长度
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        BA_length = self.distance([xa, yb], [xb, yb])
        BC_length = BA_length
        unit_dx = (xb - xa) / BA_length
        unit_dy = (yb - ya) / BA_length

        perp_dx, perp_dy = unit_dy, -unit_dx
        xc, yc = xb + BC_length * perp_dx, yb + BC_length * perp_dy
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
    
    def allocate_equilateral_triangle(self, clause):
        # 确定中垂线，延长sqrt(3) / 2 * length
        # OC = OM + MC, OM = (OA + OB) / 2
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        xm, ym =  (xa + xb) / 2, (ya + yb) / 2
        
        BA = [xb - xa, yb - ya]
        BA_length = self.distance([xa, yb], [xb, yb])
        MC_length = math.sqrt(3) / 2 * BA_length
        
        unit_dx = (xb - xa) / BA_length
        unit_dy = (yb - ya) / BA_length

        perp_dx, perp_dy = -unit_dy, unit_dx

        xc, yc = xm - MC_length * perp_dx, ym - MC_length * perp_dy
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        
    def allocate_kite(self, clause):
        # 先确定A,B,D, 用向量关系确定C
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = -math.radians(random.uniform(60, 120)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)

        xd = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yd = ya + (xb - xa) * sin_val + (yb - ya) * cos_val
        # OC = OA + AC = OA + k(AB + AD)
        if abs(top_angle) > math.radians(90):
            ratio = random.uniform(1.5, 2.0)
        else:
            ratio = random.uniform(0.75, 0.9)
        xc = xa + ((xb - xa) + (xd - xa)) * ratio
        yc = ya + ((yb - ya) + (yd - ya)) * ratio
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_parallelogram(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(random.uniform(45, 135)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        
        ratio = random.uniform(1.2, 1.5)
        xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
        yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio

        # OC = OA + AC = OA + AB + AD
        xc = xa + (xb - xa) + (xd - xa)
        yc = ya + (yb - ya) + (yd - ya)
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_rhombus(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(random.uniform(45, 135)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        
        xd = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yd = ya + (xb - xa) * sin_val + (yb - ya) * cos_val
        # OC = OA + AC = OA + AB + AD
        xc = xa + (xb - xa) + (xd - xa)
        yc = ya + (yb - ya) + (yd - ya)
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_rectangle(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(90) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        ratio = random.uniform(1.2, 1.5)
        xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
        yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio

        # OC = OA + AC = OA + AB + AD
        xc = xa + (xb - xa) + (xd - xa)
        yc = ya + (yb - ya) + (yd - ya)
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_square(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(90) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)

        xd = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yd = ya + (xb - xa) * sin_val + (yb - ya) * cos_val

        # OC = OA + AC = OA + AB + AD
        xc = xa + (xb - xa) + (xd - xa)
        yc = ya + (yb - ya) + (yd - ya)
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_trapezoid(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(random.uniform(60, 120)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        
        ratio = random.uniform(1.2, 1.5)
        xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
        yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio

        # OC = OA + AC = OA + AB + k * AD
        # 钝角对应增加长度，锐角对应减小长度
        ratio_2 = random.uniform(1.8, 2.4) if abs(top_angle) > math.pi / 2 else\
            random.uniform(0.6, 0.8)
        xc = xa + (xb - xa) + (xd - xa) * ratio_2
        yc = ya + (yb - ya) + (yd - ya) * ratio_2
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
        
    def allocate_isosceles_trapezoid(self, clause):
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(random.uniform(100, 145)) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle) 
        ratio = random.uniform(1, 1.5)
        xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
        yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio

        # OE = OA + AE = OA + AB + AD
        xe = xa + (xb - xa) + (xd - xa) 
        ye = ya + (yb - ya) + (yd - ya)
        # 将E绕D逆时针旋转2\theta-180得到C
        rotate_angle = 2*top_angle - math.pi
        cos_val_2 = math.cos(rotate_angle)
        sin_val_2 = math.sin(rotate_angle) 
        xc = xd + (xe - xd) * cos_val_2 - (ye - yd) * sin_val_2
        yc = yd + (xe - xd) * sin_val_2 + (ye - yd) * cos_val_2
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
          
    def allocate_right_trapezoid(self, clause):
        # AB 为直角边
        predicate, items = parse_clause(clause)
        A, B, C, D = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        xb, yb = self.random_allocate_position()
        top_angle = - math.radians(90) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        
        ratio = random.uniform(1.2, 1.5)
        xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
        yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio

        # OC = OA + AC = OA + AB + k * AD
        ratio_2 = random.uniform(1.2, 1.5) 
        xc = xa + (xb - xa) + (xd - xa) * ratio_2
        yc = ya + (yb - ya) + (yd - ya) * ratio_2
        
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
        self.p_pos[D] = [xd, yd]
            
    def allocate_for_relation(self, clause):
        name = get_predicate_name(clause)

        if name == 'Collinear':
            self.allocate_collinear(clause)
        elif name == 'Cocircular':
            self.allocate_cocircular(clause)
        elif name == 'Polygon':
            self.allocate_polygon(clause)
        elif name == 'Equal':
            self.allocate_equal(clause)
        elif name == 'ParallelBetweenLine':
            self.allocate_parallel(clause)
        elif name == 'IsCentroidOfTriangle':
            self.allocate_centroid(clause)
        elif name == 'IsOrthocenterOfTriangle':
            self.allocate_orthocenter(clause)
        
        elif name in ['IsCircumcenterOfTriangle', 'IsCircumcenterOfQuadrilateral']:
            self.allocate_circumcenter(clause)

        self.update_position(clause)
        
    def allocate_polygon(self, clause):
        
        predicate, items = parse_clause(clause)
        points = items[0]
        self.define_points(points)
        if len(points) == 2:
            return
        if len(points) == 3:
            self.allocate_triangle(clause)
        if len(points) == 4:
            self.allocate_quad(clause)
                        
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
    
    def allocate_congruent_similar_base(self, clause):
        predicate, items = parse_clause(clause)
        points_1, points_2 = items
        self.define_points(list(points_1 + points_2))

        if len(points_1) == 3:
            self.allocate_triangle(f"Triangle({points_1})")
        else:
            self.allocate_quad(f"Quadrilateral({points_1})")
        min_x = min([self.p_pos[p][0] for p in points_1])
        max_x = max([self.p_pos[p][0] for p in points_1])
        min_y = min([self.p_pos[p][1] for p in points_1])
        max_y = max([self.p_pos[p][1] for p in points_1])
        bbox_x = max_x - min_x
        bbox_y = max_y - min_y

        ratio = random.choice([random.uniform(0.5, 0.8), 
                               random.uniform(1.2, 1.8)]) # scale ratio
        if 'Congruent' in predicate:
            ratio = 1.0
            
        if bbox_x > bbox_y:
            offset_x = random.uniform(-3, 3)
            offset_y = random.uniform(max(1, ratio)*bbox_y, 1.25*bbox_y)
        else:
            offset_x = random.uniform(max(1, ratio)*bbox_x, 1.25*bbox_x)
            offset_y = random.uniform(-3, 3)
        
        positions_1 = [self.p_pos[p] for p in points_1]
        positions_2 = []
        xa, ya = positions_1[0]
        # 以第一个点为缩放中心，缩放ratio倍，再加offset
        for i, p in enumerate(points_2):
            if i == 0:
                positions_2.append([xa, ya])
            else:
                xi_1, yi_1 = positions_1[i]
                xi_2, yi_2 = xa + ratio * (xi_1 - xa), ya + ratio * (yi_1 - ya)
                positions_2.append([xi_2, yi_2])
        # add offset
        positions_2 = [[xi + offset_x, yi + offset_y] for [xi, yi] in positions_2]
        for p, pos in zip(points_2, positions_2):
            self.p_pos[p] = pos
        return
    
            
    def allocate_centroid(self, clause):
        predicate, items = parse_clause(clause)
        D = items[0]
        A, B, C = items[1]
        xa, ya = self.p_pos[A]
        xb, yb = self.p_pos[B]
        xc, yc = self.p_pos[C]
        xo, yo = (xa + xb + xc) / 3, (ya + yb + yc) / 3
        self.p_pos[D] = [xo, yo]
    
    def allocate_orthocenter(self, clause):
        predicate, items = parse_clause(clause)
        D = items[0]
        A, B, C = items[1]
        # AD \perp BC, BD \perp AC
        self.define_points([D])

        dot_prod_1 = self.get_dot_product([A, D], [B, C])
        dot_prod_2 = self.get_dot_product([B, D], [A, C])
        eq_1 = Eq(dot_prod_1, 0)
        eq_2 = Eq(dot_prod_2, 0)
        char, target, expand_eq = self.find_target([D])
        if target is None:
            return False

        solution = solve((eq_1, eq_2) + expand_eq, target)
        self.update_symbol(solution, char) 
        return
    
    def allocate_circumcenter(self, clause):
        predicate, items = parse_clause(clause)
        circle, points = items
        char, target, expand_eq = self.find_target([circle])
        if target is None:
            return
        circle_eqs = self.get_circum_circle_eqs(target, points[:3])
        solution = solve(circle_eqs + expand_eq, target)
        self.update_symbol(solution, circle)

    def allocate_equal_angle(self, angle_1, angle_2):
        if not angle_1.isdigit(): 
            self.define_points(angle_1)
        if not angle_2.isdigit(): # angle abc = angle def
            self.define_points(angle_2)
        
        if len(set(angle_1) & set(angle_2)) == 2 and angle_1[1] == angle_2[1]:
            # BD is besector of angle ABC, angle ABD = CBD
            B = angle_1[1]
            D = list(set(angle_1) & set(angle_2) - set([B]))[0]
            A = list(set(angle_1) - set([B, D]))[0]
            C = list(set(angle_2) - set([B, D]))[0]
            xb, yb = self.p_pos[B]
            xa, ya = self.p_pos[A]
            xc, yc = self.p_pos[C]
            xd, yd = self.p_pos[D]
            BA_length = self.get_line_length([B, A])
            BC_length = self.get_line_length([B, C])
            BA_unit = ((xa - xb) / BA_length, (ya - yb) / BA_length)
            BC_unit = ((xc - xb) / BC_length, (yc - yb) / BC_length)
            BD_vec = (BA_unit[0] + BC_unit[0], BA_unit[1] + BC_unit[1])
            angle_eq = Eq((yd - yb) / (xd - xb), BD_vec[1] / BD_vec[0])
            char, target, expand_eq = self.find_target([D])
            
            
        else:
        # get expression of cos
            cos_1 = self.get_cos_2(angle_1)
            cos_2 = self.get_cos_2(angle_2)
            if type(cos_1) in [int, float, Float] and type(cos_2) in [int, float, Float]:
                if abs(cos_1 - cos_2) < 1e-8:
                    return
                else:
                    raise ValueError(cos_1 - cos_2)
            expr = simplify_and_trim(simplify(cos_1 - cos_2))
            angle_eq = Eq(expr, 0)

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
        # if there's one shared point, AB = BC, B is on perp midseg of AC
        if len(set(line_1) & set(line_2)) == 1:
            B = list(set(line_1) & set(line_2))[0]
            A = list(set(line_1) - (set(line_1) & set(line_2)))[0]
            C = list(set(line_2) - (set(line_1) & set(line_2)))[0]
            xb, yb = self.p_pos[B]
            xa, ya = self.p_pos[A]
            xc, yc = self.p_pos[C]
            k_ac = (ya - yc) / (xa - xc) 
            k_perp_ab = -1 / k_ac
            xm, ym = (xa + xc) /2 , (ya + yc) / 2 
            eq = Eq(yb - ym, k_perp_ab * (xb - xm))   
            
        # there's no shared point, len AB = len CD
        else:
            len_1 = self.get_line_length_2(line_1)
            len_2 = self.get_line_length_2(line_2)
            if isinstance(len_1, float) and isinstance(len_2, float):
                if abs(len_1 - len_2) < 1e-8:
                    return
                else:
                    raise ValueError(len_1 - len_2)
                
            expr = simplify_and_trim(simplify(len_1 - len_2))
            eq = Eq(expr, 0)
            
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

        O, p1, p2 = arc_1
        O, p3, p4 = arc_2
        length_1 = self.get_line_length_2([p1, p2])
        length_2 = self.get_line_length_2([p3, p4])
        
        if isinstance(length_1, float) and isinstance(length_2, float):
            if abs(length_1 - length_2) < 1e-8:
                return
            else:
                raise ValueError(length_1 - length_2)
        expr = simplify_and_trim(simplify(length_1 - length_2))
        eq = Eq(expr, 0)
        
        # find the max index point as target
        if arc_2.isdigit():
            char, target, expand_eq = self.find_target(arc_1[1:])
        else:
            char, target, expand_eq = self.find_target(arc_1[1:] + arc_2[1:])
        # solve the equation
        if target is None:
            return False
        
        r_len_1 = self.get_line_length_2([O, min(p1, p2)])
        r_len_2 = self.get_line_length_2([O, char])
        r_expr = simplify_and_trim(simplify(r_len_1 - r_len_2))
        r_eq = Eq(r_expr, 0)

        # circle_eqs = self.get_circum_circle_eqs(target, points[:3])
        solution = solve((eq, r_eq) + expand_eq, target)
        # find solution with Inferior arc
        x0, y0 = self.p_pos[min(p1, p2)]
        x1, y1 = solution[0]
        x2, y2 = solution[1]
        dist_1 = (x0 - x1) ** 2 + (y0 - y1) ** 2
        dist_2 = (x0 - x2) ** 2 + (y0 - y2) ** 2
        solution = solution[0] if dist_1 < dist_2 else solution[1]
        # update position of target point
        self.update_symbol(solution, char) 
        return 
    
    
    def get_cos(self, angle):
        # angle ABC
        if angle.isdigit():
            return float(cos(float(angle) * pi / 180))
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
    
    def get_cos_2(self, angle):
        if angle.isdigit():
            return float(cos(float(angle) * pi / 180) ** 2)
        p1, p2, p3 = angle
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        xc, yc = self.p_pos[p3]
        BA = (xa - xb, ya - yb)
        BC = (xc - xb, yc - yb)
        dot_product = BA[0]*BC[0] + BA[1]*BC[1]
        len_BA_2 = (BA[0]**2 + BA[1]**2)
        len_BC_2 = (BC[0]**2 + BC[1]**2)
        return dot_product ** 2 / (len_BA_2 * len_BC_2)
    
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
        len_BA_2 = (xa - xb)**2 + (ya - yb)**2
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
            solution_ = []
            for s in solution:
                if type(s[0]) in [Float, float] and type(s[1]) in [Float, float]:
                    if abs(s[0]) > 1e5 or abs(s[1]) > 1e5:
                        continue
                solution_.append(s)
                    
            if len(solution_) == 0:
                raise ValueError(solution)
            x, y = random.choice(solution_)
        elif type(solution) == tuple:
            x, y = solution
        else:
            print('Error: ', type(solution))
            raise TypeError(solution)
        
        
        # x = simplify_and_trim(x)
        # y = simplify_and_trim(y)
        
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
                        position = self.random_allocate_position()
                        self.update_values(p, position)
        
    def random_allocate_position(self, n=10):
        # random allocate for n times
        # choose the pos that \max \min dist(p, p_i), p_i in self.p_pos
        max_distance = 0
        best_point = None

        for _ in range(n):
            x = random.uniform(5, 10)
            y = random.uniform(5, 10)
            min_distance = float('inf')

            for p, pos in self.p_pos.items():
                if pos is None or type(pos[0]) not in [float, Float]:
                    continue
                dist = self.distance_2([x, y], pos)
                if dist < min_distance:
                    min_distance = dist

            if min_distance > max_distance:
                max_distance = min_distance
                best_point = [x, y]

        return best_point
            
    def update_values(self, point, position):
        # point: char; position: float value
        x_value, y_value = position
        x_sym, y_sym = get_symbol(point)
        # export x_sym = x_value, y_sym = y_value in all expressions
        for p in self.points:
            if self.p_pos[p] is None:
                continue
            x_p, y_p = self.p_pos[p]
            if type(x_p) in [int, float, Float] and type(y_p) in [int, float, Float]:
                continue
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
            # scale for cases: x_e, 10 * x_e
            if type(x_p) in [int, float, Float]:
                if abs(x_p) > 20:
                    scale = random.uniform(10, 15) / abs(x_p)
                    x_p = x_p * scale
                    y_p = y_p * scale
            if type(y_p) in [int, float, Float]:
                if abs(y_p) > 20:
                    scale = random.uniform(10, 15) / abs(y_p)
                    x_p = x_p * scale
                    y_p = y_p * scale
            
            if isinstance(x_p, Float):
                x_p = float(x_p)
            if isinstance(y_p, Float):
                y_p = float(y_p)
            self.p_pos[p] = [x_p, y_p]

            
if __name__ == '__main__':
    setup_seed(1234)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(5):
        # clauses_base = random.choices(PREDICATES_ENT, k=1)
        # clauses_rel = random.choices(PREDICATES_REL, k=2)
        clauses_base = ['CongruentBetweenQuadrilateral']
        clauses_rel = ['IsMedianOfTriangle']
        # clauses_rel = ['IsCentroidOfTriangle', 'IsMedianOfTriangle', 'IsMidsegmentOfTriangle']
        
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_new_lines=2
        )
        print('---------- Chosen Predicates ----------')
        print('clauses_base: ', clauses_base)
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

        