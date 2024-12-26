import itertools
import math
import random
import re
import string
from copy import deepcopy

import numpy as np
import sympy
from sympy import (And, Eq, Expr, Float, I, Or, StrictGreaterThan,
                   StrictLessThan, Symbol, cos, im, nsimplify, oo, pi,
                   simplify, solve, symbols)

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils.preset import PREDICATES_ENT, PREDICATES_REL
from utils.symbolic import (find_target_for_construct, get_predicate_name,
                            get_symbol, max_letter_index, parse_clause,
                            random_generate_angle_measure,
                            random_generate_line_length, replace_for_clause,
                            replace_points, simplify_and_trim)
from utils.tools import append_lst, remove_duplicates, setup_seed

IGNORE_NAMES = [
    'Shape',
    'IsCircumcenterOfQuadrilateral',
    'IsCircumcenterOfTriangle',
    
]
    
def get_max_letter(s):
    return sorted(s)[-1]
    

class Allocator():
    def __init__(self, 
                 geo_states, 
                 construct_cdls, 
                 text_cdls,
                 allocate_value=False,
                 replace_chars=False):
        self.ori_states = {
            "geo_states": deepcopy(geo_states),
            "construct_cdls": deepcopy(construct_cdls),
            "text_cdls": deepcopy(text_cdls)
        }
        self.points = geo_states['points']                     
        self.lines = geo_states['lines']
        self.circles = geo_states['circles']
        assert len(self.circles) in [0, 1, 2]
        self.polygons = geo_states['polygons']
        
        self.constraints = [c for c in geo_states['constraints'] 
                            if c not in geo_states['constraints_base']]
        self.constraints_base = geo_states['constraints_base']
        self.points_on_circle = geo_states['points_on_circle']
        self.construct_cdls = construct_cdls
        self.text_cdls = text_cdls
        self.image_cdls = []
        
        self.clauses = self.construct_cdls + self.constraints
        # merge all cocircular clauses
        self.clauses = self.merge_cocircular_clauses()
        # sort by idx of points
        # [max_letter_index(x) for x in self.clauses]
        self.clauses = sorted(list(set(self.clauses)), key=max_letter_index)
        # filter by IGNORE_NAMES
        self.clauses = [c for c in self.clauses if c.split('(')[0] not in IGNORE_NAMES]

        # store the location of points
        self.p_pos = {p: None for p in self.points}
        self.p_pos['a'] = [0, 0]
        self.p_pos_range = {}

        # find min clauses subset needed to construct each point 
        # key: point | value: subset of clauses
        # if all clauses are visited in one subset
        # then random allocate position for this point
        self.clause_subset = self.find_mini_clauses_subset()
        self._formulated_cdls = None
        
        # whether to allocate line length or angle measre
        self.allocate_value = allocate_value
        # whether to replace template abc -> upper chars
        self.replace_chars = replace_chars
        
    @property
    def states(self):
        return {
            "p_pos": self.p_pos,
            "points": self.points,
            "lines": self.lines,
            "circles": self.circles,
            "points_on_circle": self.points_on_circle,
            "clauses": self.clauses,
            "clauses_base": self.constraints_base
        }
        
    @property
    def formulated_cdls(self):
        if self._formulated_cdls is None:
            construct_cdls = []
            collinear_cdls = []
            cocircular_cdls = []
            for poly in self.polygons:
                lines = []
                for i in range(len(poly)):
                    lines.append(f"{poly[i]}{poly[(i+1) % len(poly)]}")
                poly_cdl = f"Shape({','.join(lines)})"
                construct_cdls.append(poly_cdl)
                
            for const_cdl in self.construct_cdls:
                if 'Collinear' in const_cdl:
                    collinear_cdls.append(const_cdl)
                if 'Cocircular' in const_cdl:
                    cocircular_cdls.append(const_cdl)
                    
            if len(collinear_cdls) > 0:
                for cdl in collinear_cdls:
                    _, items = parse_clause(cdl)
                    # sort by coords x
                    points = sorted(items[0], key=lambda x: self.p_pos[x][0])
                    collinear_cdl = f"Collinear({''.join(points)})"
                    construct_cdls.append(collinear_cdl)
            
            if len(cocircular_cdls) > 0:
                points_on_circle = {}
                for cdl in cocircular_cdls:
                    _, items = parse_clause(cdl)
                    circle, ps = items
                    if circle not in points_on_circle:
                        points_on_circle[circle] = set()
                    points_on_circle[circle].update(ps)
                for circle, ps in points_on_circle.items():
                    # sort by counter clockwise
                    sorted_ps = self.sort_points_counter_clockwise(list(ps))
                    cocircular_cdl = f"Cocircular({circle},{''.join(sorted_ps)})"
                    construct_cdls.append(cocircular_cdl)
            
            
            text_cdls = remove_duplicates(self.text_cdls)
            construct_cdls = remove_duplicates(construct_cdls)
            if len(self.image_cdls) == 0:
                self.find_image_cdls()
            self._formulated_cdls = {
                "text_cdls": text_cdls,
                "construct_cdls": construct_cdls,
                "image_cdls": self.image_cdls
            }
        return self._formulated_cdls
        
    def empty_states(self):
        geo_states = self.ori_states['geo_states']

        self.points = geo_states['points']                     
        self.lines = geo_states['lines']
        self.circles = geo_states['circles']
        self.polygons = geo_states['polygons']
        
        self.constraints = [c for c in geo_states['constraints'] 
                            if c not in geo_states['constraints_base']]
        self.constraints_base = geo_states['constraints_base']
        self.points_on_circle = geo_states['points_on_circle']
        self.construct_cdls = self.ori_states['construct_cdls']
        self.text_cdls = self.ori_states['text_cdls']
        self.image_cdls = []
        
        self.clauses = self.construct_cdls + self.constraints
        # merge all cocircular clauses
        self.clauses = self.merge_cocircular_clauses()
        # sort by idx of points
        # [max_letter_index(x) for x in self.clauses]
        self.clauses = sorted(list(set(self.clauses)), key=max_letter_index)
        # filter by IGNORE_NAMES
        self.clauses = [c for c in self.clauses if c.split('(')[0] not in IGNORE_NAMES]

        self.p_pos = {p: None for p in self.points}
        self.p_pos['a'] = [0, 0]
        self.p_pos_range = {}
        self.clause_subset = self.find_mini_clauses_subset()
        self._formulated_cdls = None
        
    @staticmethod
    def distance_2(point1, point2):
        return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            
    def allocate(self):
        # loop until get in-complex solution
        for i in range(5):
            self.empty_states()
            self.allocate_for_base(self.text_cdls[0])
            
            for clause in self.clauses:
                self.allocate_for_relation(clause)

            if self.check_complex():
                continue

            # if there're still point remains unconstrained:
            # use collinear or cocircular to constrain it
            for p in self.points:
                if self.p_pos[p] is None:
                    position = self.random_allocate_position()
                    self.update_values(p, position)
                else:
                    x, y = self.p_pos[p]
                    if type(x) not in [Float, float, int] or type(y) not in [Float, float, int]:
                        # try cocircular first, then collinear
                        position = None
                        if len(self.circles) > 0:
                            position = self.find_possible_cocircular(p)
                        if position is None:
                            position = self.find_possible_collinear(p)
                        if position is None:
                            position = self.random_allocate_position(target_p=p)
                        self.update_values(p, position)
            
            if self.check_complex():
                continue
            
            # check if there's unsolved point
            allocate_success = True
            for p in self.points:
                if self.p_pos[p] is None:
                    allocate_success = False
                else:
                    x, y = self.p_pos[p]
                    if type(x) not in [Float, float, int] or type(y) not in [Float, float, int]:
                        allocate_success = False
            if not allocate_success:
                continue
                
            # check if out of figure size
            positions = list(self.p_pos.values())
            try:
                min_x = np.min([pos[0] for pos in positions])
                max_x = np.max([pos[0] for pos in positions])
                min_y = np.min([pos[1] for pos in positions])
                max_y = np.max([pos[1] for pos in positions])
            except Exception as e:
                print(e)
                raise e
            max_val = max([abs(i) for i in [min_x, max_x, min_y, max_y]])
            if max_val > 80:
                continue
            
            
            # clauses new added
            self.clauses = sorted(list(set(self.clauses)), key=max_letter_index)
            
            # delete coincide points and modify clauses
            self.delete_coincide_points()
            
            # find triangles and quads, add 'Shape($)' in construct cdls 
            self.find_triangles()
            self.find_quads()
            self.find_image_cdls()
            
            # allocate value of line or angle, random choose
            if self.allocate_value:
                self.allocate_for_value()

            self.replace_characters()
            return
        
        print('Fail to allocate positions.')
        
    def replace_characters(self):
        if self.replace_chars:
            start_char = random.choice(['A', 'E', 'I', 'M', 'R'])
        else:
            start_char = 'A'
        start_idx = string.ascii_uppercase.index(start_char)
        chars = string.ascii_uppercase[start_idx: start_idx + len(self.p_pos)]
        mapping = {}
        for point_i, char in zip(self.p_pos, chars):
            mapping[point_i] = char
            
        # points
        self.points = [mapping[p] for p in self.points]
        self.p_pos = {mapping[k]: self.p_pos[k] for k in self.p_pos}
        # lines / circles / polygons
        self.lines = [tuple([mapping[p] for p in l]) for l in self.lines]
        self.circles = [mapping[p] for p in self.circles]
        self.points_on_circle = {
            mapping[k]: [mapping[x] for x in v] 
            for k, v in self.points_on_circle.items()
        }
        self.polygons = [tuple([mapping[c] for c in p]) for p in self.polygons]
        
        # clauses
        self.clauses = [replace_for_clause(c, mapping) for c in self.clauses]
        self.text_cdls = [replace_for_clause(c, mapping) for c in self.text_cdls]
        self.construct_cdls = [replace_for_clause(c, mapping) for c in self.construct_cdls]
        self.image_cdls = [replace_for_clause(c, mapping)for c in self.image_cdls]
        return
    
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
    
    def check_complex(self):
        # check if there's complex solution: a + bI
        have_complex = False
        for p, pos in self.p_pos.items():
            if type(pos[0]) in [Float, float, int]:
                continue
            x, y = pos
            if x.has(I) or y.has(I):
                have_complex = True
                
        if have_complex:
            print('Get complex solution. ')
            self.empty_states()
            return True
        return False
        
    def find_possible_collinear(self, target_p):
        # get unsolved point with possible collinear constraint
        possible_lines = [l for l in self.lines if target_p not in l]
        random.shuffle(possible_lines)
        for line in possible_lines:
            x, y = self.p_pos[target_p]
            x1, y1 = self.p_pos[line[0]]
            x2, y2 = self.p_pos[line[1]]
            if not all([type(i) in [float, Float, int] for i in [x1, y1, x2, y2]]):
                continue
            # (y2-y1)x - (x2-x1)y + (x2y1-x1y2) = 0
            expr = (y2 - y1) * x - (x2 - x1) * y + (x2 * y1 - x1 * y2)
            expr = simplify_and_trim(expr)
            line_eq = Eq(expr, 0)
            if line_eq == True or line_eq == False:
                continue
            _, target, expand_eq = self.find_target([target_p])
            solution = solve((line_eq, ) + expand_eq, target)
            res = self.update_symbol(solution, target_p)
            if res is not None:
                x, y = res
                zipped_pairs = zip([x1, x2, x], [line[0], line[1], target_p])
                sorted_pairs = sorted(zipped_pairs)
                _, sorted_ps = zip(*sorted_pairs)
                sorted_ps = list(sorted_ps)
                # add construct cdls
                self.construct_cdls.append(f"Collinear({''.join(sorted_ps)})")
                # add new lines
                for (i, j) in list(itertools.combinations(sorted_ps, 2)):
                    if tuple([i, j]) in self.lines:
                        self.lines.remove(tuple([i, j]))
                self.lines.append(tuple(sorted_ps))
                return [x, y]
            
        return None
    
    def find_possible_cocircular(self, target_p):
        for c in self.circles:
            if target_p in self.points_on_circle[c]:
                return None
        circle = random.choice(self.circles)
        xc, yc = self.p_pos[circle]
        x0, y0 = self.p_pos[self.points_on_circle[circle][0]] # first point on circle
        x, y = self.p_pos[target_p]
        dist_1 = (x0 - xc)**2 + (y0 - yc)**2
        dist_2 = (x - xc)**2 + (y - yc)**2
        
        expr = simplify_and_trim(dist_1 - dist_2)
        line_eq = Eq(expr, 0)
        _, target, expand_eq = self.find_target([target_p])
        solution = solve((line_eq, ) + expand_eq, target)
        
        res = self.update_symbol(solution, target_p)
        if res is not None:
            x, y = res
            cocircular_ps = sorted(self.points_on_circle[circle] + [target_p])
            self.construct_cdls.append(f"Cocircular({circle},{''.join(cocircular_ps)})")
            # add new ps on circle
            self.points_on_circle[circle] = cocircular_ps
            return [x, y]
                
        return None

    def sort_points_counter_clockwise(self, points):
        if len(points) < 3:
            return points
        
        start_point = points[0]
        # calcualte center coordinate
        x_coords = [self.p_pos[p][0] for p in points]
        y_coords = [self.p_pos[p][1] for p in points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        def calculate_angle(p):
            dx = self.p_pos[p][0] - center_x
            dy = self.p_pos[p][1] - center_y
            return math.atan2(dy, dx)

        # sort by calculated angle, reverse=True to reverse for cv2
        sorted_points = sorted(points, 
                               key=lambda x: calculate_angle(x), 
                               reverse=True)

        # adjust to start with initial point
        start_index = sorted_points.index(start_point)
        sorted_points = sorted_points[start_index:] + sorted_points[:start_index]

        return sorted_points


    def find_triangles(self):
        # delete all triangles first
        polygons_ = []
        for poly in self.polygons:
            if len(poly) != 3:
                polygons_.append(poly)
                
        possible_triangles = list(itertools.permutations(self.points, 3))
        triangles = []
        for points in possible_triangles:
            l1 = tuple([points[0], points[1]])
            l2 = tuple([points[1], points[2]])
            l3 = tuple([points[0], points[2]])
            l1_exist, l2_exist, l3_exist = False, False, False
            collinear = False
            for l in self.lines:
                if all([p in l for p in l1]):
                    l1_exist = True
                if all([p in l for p in l2]):
                    l2_exist = True
                if all([p in l for p in l3]):
                    l3_exist = True
                
            cos_val = self.get_cos_2(''.join(points))
            if abs(abs(cos_val) - 1) < 1e-2:
                collinear = True
            # AB x AC > 0 (in cv2 < 0)
            xa, ya = self.p_pos[points[0]]
            xb, yb = self.p_pos[points[1]]
            xc, yc = self.p_pos[points[2]]
            AB = [xb - xa, yb - ya]
            AC = [xc - xa, yc - ya]
            counter_clock_wise = AB[0] * AC[1] - AB[1] * AC[0] < 0
            if all([l1_exist, l2_exist, l3_exist, counter_clock_wise]) and not collinear:
                triangles.append(points)
        
        # delete duplicate quads
        seen = set()
        triangles = tuple([q for q in triangles if not (tuple(sorted(q)) in seen or seen.add(tuple(sorted(q))))])  
        self.polygons = append_lst(polygons_, triangles)
        
    def find_quads(self):
        # delete all quads first
        # different from triangle: check counter clock wise
        polygons_ = []
        for poly in self.polygons:
            if len(poly) != 4:
                polygons_.append(poly)
                
        possible_quads = list(itertools.permutations(self.points, 4))
        quads = []
        for points in possible_quads:
            # line must exist
            l_exist = [False for i in range(4)]
            l_list = [tuple([points[i], points[(i+1)%4]]) for i in range(4)]
            for idx, li in enumerate(l_list):
                for l in self.lines:
                    if all([p in l for p in li]):
                        l_exist[idx] = True
                        break
            
            # cannot collinear in pairs
            collinear = False
            for i in range(4):
                l1, l2 = l_list[i], l_list[(i+1)%4]
                for l in self.lines:
                    if all([p in l for p in l1+l2]):
                        collinear = True
                        break
                    
            # AB x AC > 0, AC x AD > 0 (in cv2 < 0)
            xa, ya = self.p_pos[points[0]]
            xb, yb = self.p_pos[points[1]]
            xc, yc = self.p_pos[points[2]]
            xd, yd = self.p_pos[points[3]]
            AB = [xb - xa, yb - ya]
            AC = [xc - xa, yc - ya]
            AD = [xd - xa, yd - ya]
            counter_clock_wise_1 = AB[0] * AC[1] - AB[1] * AC[0] < 0
            counter_clock_wise_2 = AC[0] * AD[1] - AC[1] * AD[0] < 0
            
            if all(l_exist + [counter_clock_wise_1, counter_clock_wise_2]) and not collinear:
                quads.append(points)
                
        # delete duplicate quads
        seen = set()
        quads = tuple([q for q in quads if not (tuple(sorted(q)) in seen or seen.add(tuple(sorted(q))))])
        
        self.polygons = append_lst(polygons_, quads)
        
    def find_image_cdls(self):
        for clause in self.clauses:
            if 'Equal' in clause:
                if any([pred in clause for pred in 
                        ['LengthOfLine', 'MeasureOfAngle', 'LengthOfArc']]):
                    if clause not in self.image_cdls:
                        self.image_cdls.append(clause)

    def allocate_for_value(self):
        # allocate value for random line or angle
        mode = random.choice([0, 1, 2])

        added_cdls = []
        if mode == 0:
            line = random.choice(self.lines)
            if len(line) > 2:
                line = line[:2]
            
            length = random_generate_line_length()
            if length != '0':
                added_cdls.append(f"Equal(LengthOfLine({''.join(line)}),{length})")
        
        elif mode == 1:
            for i in range(5): # try 5 times
                poly = random.choice(self.polygons)
                idx = random.choice(list(range(len(poly))))
                p1 = poly[idx]
                p_mid = poly[(idx+1) % len(poly)]
                p2 = poly[(idx+2) % len(poly)]
                # find angle that hasn't been allocated
                for cdl in self.image_cdls:
                    name, items = parse_clause(cdl)
                    if name == 'MaeasureOfAngle':
                        if p_mid == items[0][1] and p1 in items[0] and p2 in items[0]:
                            continue
                measure = random_generate_angle_measure(
                    self.p_pos[p_mid], self.p_pos[p1], self.p_pos[p2])
                if measure != '0' and measure != '180':
                    added_cdls.append(f"Equal(MeasureOfAngle({''.join([p1, p_mid, p2])}),{measure})")
                break
        else:
            pass
        
        self.text_cdls += added_cdls
        self.image_cdls += added_cdls

    def get_coincide_points(self, x, y, ignore_p=None):
        # find coincide points for target_p, return [] if there's no coincide
        coincide_ps = []
        for p, pos in self.p_pos.items():
            if ignore_p != None and p == ignore_p:
                continue
            if pos == None or type(pos[0]) not in [float, Float, int]:
                continue
            dist = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            if dist < 0.01:
                coincide_ps.append(p)
                
        return coincide_ps
    
    def delete_coincide_points(self):
        # delete coincide points
        points = list(self.p_pos.keys())
        positions = list(self.p_pos.values())
        mapping = {}
        for p, pos in zip(points, positions):
            if pos == None or type(pos[0]) not in [float, Float, int]:
                continue
            coincide_ps = self.get_coincide_points(pos[0], pos[1], ignore_p=p)
            if len(coincide_ps) == 0:
                mapping[p] = p
            else:
                for coincide_p in coincide_ps:
                   if coincide_p > p: # p1 -> p2, p1 > p2
                       mapping[p] = p
                       mapping[coincide_p] = p
        
        # replace chars in clauses, may delete some clause
        text_cdls = [self.replace_points_for_clause(cdl, mapping) 
                     for cdl in self.text_cdls]
        text_cdls = [c for c in text_cdls if c != None]
        construct_cdls = [self.replace_points_for_clause(cdl, mapping) 
                          for cdl in self.construct_cdls]
        construct_cdls = [c for c in construct_cdls if c != None]
        clauses = [self.replace_points_for_clause(cdl, mapping) 
                   for cdl in self.clauses]
        clauses = [c for c in clauses if c != None]
        clauses = remove_duplicates(clauses)
        self.text_cdls = text_cdls
        self.construct_cdls = construct_cdls
        self.clauses = clauses
        
        # replace chars in points, lines, circles, p_pos
        for p1, p2 in mapping.items():
            if p1 != p2 : # p1 -> p2
                # points
                self.p_pos.pop(p1, None)
                if p1 in self.points:
                    self.points.remove(p1)
                # lines
                new_lines = []
                for l in self.lines:
                    if p1 in l:
                        # replace points
                        new_line = tuple(p2 if p == p1 else p for p in l)
                        # delete repeated
                        seen = set()
                        new_line = tuple([p for p in new_line if not (p in seen or seen.add(p))])
                        if len(new_line) > 1:
                            new_lines.append(new_line)
                    else:
                        new_lines.append(l)
                self.lines = sorted(set(new_lines))
                # points on circle
                for c, ps in self.points_on_circle.items():
                    if p1 in ps:
                        ps.remove(p1)
                        if p2 not in ps:
                            ps.append(p2)
                    self.points_on_circle[c] = ps
                # circle
                if p1 in self.circles:
                    self.circles.remove(p1)
                    self.circles.append(p2)
                    self.points_on_circle[p2] = self.points_on_circle.pop(p1)
                # polygons
                new_polygons = []
                for poly in self.polygons:
                    if p1 in poly:
                        # replace points
                        new_poly = tuple(p2 if p == p1 else p for p in l)
                        # delete repeated
                        seen = set()
                        new_poly = tuple([p for p in new_poly if not (p in seen or seen.add(p))])
                        if len(new_poly) > 2:
                            new_polygons.append(new_poly)
                    else:
                        new_polygons.append(poly)
                self.polygons = sorted(set(new_polygons))
        
    def replace_points_for_clause(self, clause, mapping):
        if 'Equal' in clause:
            items = clause.split('Equal(')[-1][:-1]
            left, right = items.split(',')
            left = self.replace_points_for_clause(left, mapping)
            right = self.replace_points_for_clause(right, mapping)
            return f'Equal({left},{right})'
        elif '(' not in clause and ')' not in clause:
            return clause
        else:
            predicate, items = parse_clause(clause)
            items_res = []
            for letters in items:
                seen = set()
                # replace points
                letters = [mapping.get(c, c) for c in letters]
                letters = [x for x in letters if not (x in seen or seen.add(x))]
                items_res.append(''.join(letters))
            if predicate == 'Collinear' and len(items_res[0]) <= 2:
                return None
            clause_res = f"{predicate}({','.join(items_res)})"
            return clause_res
            
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
            mode = random.choice([0, 1, 2])
            if mode == 0:
                top_angle = - math.radians(random.uniform(45, 75)) 
            elif mode == 1:
                top_angle = - math.radians(30) 
                # must be counter clockwise, ABC -> CAB
                new_cdl = f"Equal(MeasureOfAngle({''.join([C,A,B])}),30)"
                self.image_cdls.append(new_cdl)
                self.text_cdls.append(new_cdl)
            elif mode == 2:
                top_angle = - math.radians(60) 
                new_cdl = f"Equal(MeasureOfAngle({''.join([C,A,B])}),60)"
                self.image_cdls.append(new_cdl)
                self.text_cdls.append(new_cdl)
            interval = random.choice([(0.7, 0.9), (1.1, 1.5)])
            ratio = random.uniform(interval[0], interval[1])
            cos_val = math.cos(top_angle)
            sin_val = math.sin(top_angle)
            xc = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio
            yc = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio
            self.p_pos[C] = [xc, yc]
            
    
    def allocate_quad(self, clause):
        _, items = parse_clause(clause)
        A, B, C, D = items[0]
        
        if self.p_pos[A] is None or type(self.p_pos[A][0]) not in [float, Float, int]:
            xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
            self.p_pos[A] = [xa, ya]
            
        if self.p_pos[B] is None or type(self.p_pos[B][0]) not in [float, Float, int]:
            xb, yb = self.random_allocate_position()
            self.p_pos[B] = [xb, yb]
            
        xa, ya = self.p_pos[A]
        xb, yb = self.p_pos[B]
        
        if self.p_pos[C] is None or type(self.p_pos[C][0]) not in [float, Float, int]:
            angle_BAC = - math.radians(random.uniform(30, 60)) 
            ratio_AC = random.uniform(1.1, 1.4)
            cos_val = math.cos(angle_BAC)
            sin_val = math.sin(angle_BAC)
            xc = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio_AC
            yc = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio_AC
            self.p_pos[C] = [xc, yc]
        
        xc, yc = self.p_pos[C]
        
        if self.p_pos[D] is None or type(self.p_pos[D][0]) not in [float, Float, int]:
            mode = random.choice(['random', 'perp'])
            # mode = 'cocircular'
            # mode = 'perp'
            # self.define_points([D])
                    
            if mode == 'random':
                xd, yd = self.p_pos[D] # symbol
                BAC_val = math.acos(self.get_cos(''.join((B, A, C)))) / np.pi * 180
                if BAC_val > 150:
                    angle_BAD = - math.radians(random.uniform(BAC_val, 170))
                else:
                     angle_BAD = - math.radians(random.uniform(BAC_val, 150))
                cos_val = math.cos(angle_BAD)
                sin_val = math.sin(angle_BAD)
                ratio_AD = random.uniform(0.6, 1.0)
                xd = xa + ((xb - xa) * cos_val - (yb - ya) * sin_val) * ratio_AD
                yd = ya + ((xb - xa) * sin_val + (yb - ya) * cos_val) * ratio_AD
                xd, yd = xd * ratio_AD, yd * ratio_AD
                self.p_pos[D] = [xd, yd]
                
            # angle DAC = 90, or angle DCA = 90, or DAB = 90
            if mode == 'perp':
                angle = random.choice([
                    [D, A, C], [D, C, A], [D, A, B]
                ])
                new_clause = f"Equal(MeasureOfAngle({''.join(angle)}),90)"
                new_text_clause = f"PerpendicularBetweenLine({''.join([angle[0], angle[1]])},{''.join([angle[2], angle[1]])})"
                self.text_cdls.append(new_text_clause)
                
                self.allocate_equal(new_clause)

                # AC x AD must > 0 (for cv2, AC x AD < 0)
                xd, yd = self.p_pos[D]
                A = [xc - xa, yc - ya]
                B = [xd - xa, yd - ya]
                expr = A[0] * B[1] - A[1] * B[0]
                inequality = expr < 0
                solution_ineq_1 = None
                if len(inequality.free_symbols) > 0:
                    solution_ineq_1 = solve(inequality, inequality.free_symbols.pop())
                
                # D is on the left of BC, BC x BD > 0 (for cv2 < 0)
                A = [xc - xb, yc - yb]
                B = [xd - xb, yd - yb]
                expr = A[0] * B[1] - A[1] * B[0]
                inequality = expr < 0
                solution_ineq_2 = None
                if len(inequality.free_symbols) > 0: 
                    solution_ineq_2 = solve(inequality, inequality.free_symbols.pop())
                    
                if solution_ineq_1 is not None and solution_ineq_2 is not None:
                    self.p_pos_range[D] = simplify(solution_ineq_1 & solution_ineq_2)
                elif solution_ineq_1 is not None:
                    self.p_pos_range[D] = simplify(solution_ineq_1)
                elif solution_ineq_2 is not None:
                    self.p_pos_range[D] = simplify(solution_ineq_2)
                else:
                    pass

                position = self.random_allocate_position(target_p=D)
                self.update_values(D, position)

    def allocate_right_triangle(self, clause):
        # 确定垂直单位向量，随机长度
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(85, 95)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
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
        # 将一点旋转随机角度, AB=AC
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(130, 140)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)

        interval = random.choice([(45, 80), (100, 130)])
        top_angle = - math.radians(random.uniform(interval[0], interval[1])) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        xc = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yc = ya + (xb - xa) * sin_val + (yb - ya) * cos_val
        self.p_pos[A] = [xa, ya]
        self.p_pos[B] = [xb, yb]
        self.p_pos[C] = [xc, yc]
            
    def allocate_isosceles_right_triangle(self, clause):
        # 确定垂直单位向量，固定长度, AB = AC, AB \perp AC
        predicate, items = parse_clause(clause)
        A, B, C = items[0]
        xa, ya = self.random_allocate_position() if A != 'a' else self.p_pos[A]
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(130, 140)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
        top_angle = -math.radians(90) 
        cos_val = math.cos(top_angle)
        sin_val = math.sin(top_angle)
        xc = xa + (xb - xa) * cos_val - (yb - ya) * sin_val
        yc = ya + (xb - xa) * sin_val + (yb - ya) * cos_val
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
        interval = random.choice([(50, 80), (100, 130)])
        top_angle = -math.radians(random.uniform(interval[0], interval[1])) 
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
        interval = random.choice([(45, 80), (100, 135)])
        top_angle = - math.radians(random.uniform(interval[0], interval[1])) 
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
        interval = random.choice([(45, 80), (100, 135)])
        top_angle = - math.radians(random.uniform(interval[0], interval[1]))
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
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(85, 95)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
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
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(85, 95)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
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
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(90, 120)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
        interval = random.choice([(60, 80), (100, 120)])
        top_angle = - math.radians(random.uniform(interval[0], interval[1])) 
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
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(90, 120)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
        top_angle = - math.radians(random.uniform(100, 135)) 
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
        type_flag = random.choice([0, 1])
        if type_flag == 0:
            xb, yb = self.random_allocate_position()
        else:
            theta = math.radians(random.uniform(85, 95)) 
            b_len = random.uniform(5, 10)
            xb, yb = b_len * math.cos(theta), b_len * math.sin(theta)
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
            return
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
        
        expr = (y2 - y1) * (x4 - x3) - (x2 - x1) * (y4 - y3)
        expr = simplify_and_trim(expr)
        eq = Eq(expr, 0)
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
            self.update_symbol(solution, other_p, cannot_coincide=False)
        
        return
    
    def allocate_cocircular(self, clause):
        predicate, items = parse_clause(clause)
        circle, points = items[0], items[1]
        self.define_points([circle] + list(points))
        
        if circle > max(points[:3]): # define new circle center
            char, target, expand_eq = self.find_target([circle])
            if target is None:
                return
            circle_eqs = self.get_circum_circle_eqs(target, points[:3])
            solution = solve(circle_eqs + expand_eq, target)
            self.update_symbol(solution, circle, cannot_coincide=False)
            
            # define other points (if) on circle
            r_len_2 = self.get_line_length_2([circle, points[0]])
            for p in points[3:]:
                r_len_i_2 = self.get_line_length_2([circle, p])
                eq = Eq(r_len_2, r_len_i_2)
                char, target, expand_eq = self.find_target([p])
                solution = solve((eq, ) + expand_eq, target)
                self.update_symbol(solution, p, cannot_coincide=False)
        else: 
            # define other points (if) on circle
            sorted_points = sorted(points)
            r_len = self.get_line_length_2([circle, sorted_points[0]])
            for p in sorted_points[1:]:
                r_len_i = self.get_line_length_2([circle, p])
                expr = simplify_and_trim(r_len_i - r_len)
                eq = Eq(expr, 0)
                char, target, expand_eq = self.find_target([p])
                if target is None:
                    continue
                solution = solve((eq, ) + expand_eq, target)
                solution_inequal = []
                if len(solution) == 2: 
                    # Polygon ABCD, AC x AD must > 0
                    # for cv2, Adopting clockwise direction, AC x AD < 0
                    quads = [p for p in self.polygons if len(p) == 4 and char in p]
                    if len(quads) != 0:
                        for sol in solution:
                            xa, ya = self.p_pos[quads[0][0]]
                            xc, yc = self.p_pos[quads[0][2]]
                            xd, yd = sol
                            A = [xc - xa, yc - ya]
                            B = [xd - xa, yd - ya]
                            expr = A[0] * B[1] - A[1] * B[0]
                            inequality = expr < 0
                            solution_i = solve((inequality, ) + expand_eq, target)
                            solution_inequal.append(solution_i)

                    
                self.update_symbol(solution, p, constraint=solution_inequal, cannot_coincide=False)
         
        return
    
    def allocate_congruent_similar_base(self, clause):
        predicate, items = parse_clause(clause)
        points_1, points_2 = items
        self.define_points(list(points_1 + points_2))

        if len(points_1) == 3:
            self.allocate_triangle(f"Polygon({points_1})")
        else:
            self.allocate_quad(f"Polygon({points_1})")
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
            offset_y = random.uniform(max(1.2, ratio)*bbox_y, 1.25*bbox_y)
        else:
            offset_x = random.uniform(max(1.2, ratio)*bbox_x, 1.25*bbox_x)
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
        self.update_symbol(solution, char, cannot_coincide=False) 
        return
    
    def allocate_circumcenter(self, clause):
        predicate, items = parse_clause(clause)
        circle, points = items
        char, target, expand_eq = self.find_target([circle])
        if target is None:
            return
        circle_eqs = self.get_circum_circle_eqs(target, points[:3])
        solution = solve(circle_eqs + expand_eq, target)
        self.update_symbol(solution, circle, cannot_coincide=False)

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
            
            # BD_vec[1] and yd - yb must be same positive / negative
            inequality = (yd - yb) * BD_vec[1] > 0
            # ignore_value = None
        else:
            if angle_2 == '90':
                # angle_1 ABC = 90, AB \\perp BC
                A, B, C = angle_1
                xa, ya = self.p_pos[A]
                xb, yb = self.p_pos[B]
                xc, yc = self.p_pos[C]
                dot_product = (xb - xa) * (xc - xb) + (yb - ya) * (yc - yb)
                expr = simplify_and_trim(dot_product)
                angle_eq = Eq(expr, 0)

                # check AB \\perp collinear ps in BC or AB
                if type(xb) not in [Float, float, int]:
                    collinear_ps = [l for l in self.lines if B in l and C in l and len(l) > 2]
                    if len(collinear_ps) > 0:
                        D = list(set(collinear_ps[0]) - set([B, C]))[0]
                        self.define_points([D])
                        xd, yd = self.p_pos[D]
                        # AB \\perp DC
                        dot_product = (xb - xa) * (xc - xd) + (yb - ya) * (yc - yd)
                        expr = simplify_and_trim(dot_product)
                        angle_eq = Eq(expr, 0)
                        
                    collinear_ps = [l for l in self.lines if B in l and A in l and len(l) > 2]
                    if len(collinear_ps) > 0:
                        D = list(set(collinear_ps[0]) - set([B, C]))[0]
                        self.define_points([D])
                        xd, yd = self.p_pos[D]
                        # AD \\perp BC
                        dot_product = (xd - xa) * (xc - xb) + (yd - ya) * (yc - yb)
                        expr = simplify_and_trim(dot_product)
                        angle_eq = Eq(expr, 0)
                     
                # A, B, C can't be coinside
                # ignore_value = [(xa, ya)]
                # if xb in [float, Float, int] and yb in [float, Float, int]:
                #     ignore_value.append((xb, yb))
                # if xc in [float, Float, int] and yc in [float, Float, int]:
                #     ignore_value.append((xc, yc))
            
            else:
                # get expression of cos
                cos_1 = self.get_cos_2(angle_1)
                cos_2 = self.get_cos_2(angle_2)
                if type(cos_1) in [int, float, Float] and type(cos_2) in [int, float, Float]:
                    if abs(cos_1 - cos_2) < 1e-8:
                        return
                    else:
                        raise ValueError(cos_1 - cos_2)
                expr = simplify_and_trim(cos_1 - cos_2)
                angle_eq = Eq(expr, 0)
                # ignore_value = None

            # find the max index point as target
            if angle_2.isdigit():
                char, target, expand_eq = self.find_target(angle_1)
            else:
                char, target, expand_eq = self.find_target(angle_1 + angle_2)
            inequality = None
            if target is None:
                return 
        
        solution = solve((angle_eq, ) + expand_eq, target)
        solution_inequal = [] if inequality is None else \
            [solve(inequality, target)]
        # update position of target point
        self.update_symbol(solution, char, 
                           constraint=solution_inequal, 
                           cannot_coincide=False) 
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
                
            expr = simplify_and_trim(len_1 - len_2)
            eq = Eq(expr, 0)
            
        char, target, expand_eq = self.find_target(line_1 + line_2)
        if target is None:
            return False

        solution = solve((eq, ) + expand_eq, target)
        self.update_symbol(solution, char, cannot_coincide=False) 
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
        expr = simplify_and_trim(length_1 - length_2)
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
        r_expr = simplify_and_trim(r_len_1 - r_len_2)
        r_eq = Eq(r_expr, 0)
        
        inequality = None
        # find solution with Inferior arc, arc AC = BC, angle ACB > 90
        # if len(set(arc_1[1:]) & set(arc_2[1:])) == 1:
        #     C = list(set(arc_1[1:]) & set(arc_2[1:]))[0]
        #     A = list(set(arc_1[1:]) - set([C]))[0]
        #     B = list(set(arc_2[1:]) - set([C]))[0]
        #     # CB dot CA < 0
        #     expr = self.get_dot_product([C, A], [C, B])
        #     inequality = expr <= 0 
        x0, y0 = self.p_pos[min(p1, p2)]
        

        solution = solve((eq, r_eq) + expand_eq, target)
        # solution = solve((eq, ) + expand_eq, target)
        solution_inequal = [] if inequality is None else \
            [solve((inequality, ) + expand_eq, target)]
        
        if len(solution) == 2:
            x1, y1 = solution[0]
            x2, y2 = solution[1]
            if type(x1) in [Float, float] and type(x2) in [Float, float]:
                dist_1 = (x0 - x1) ** 2 + (y0 - y1) ** 2
                dist_2 = (x0 - x2) ** 2 + (y0 - y2) ** 2
                solution = solution[0] if dist_1 < dist_2 else solution[1]
        # update position of target point
        self.update_symbol(solution, char, constraint=solution_inequal, cannot_coincide=False) 
        return 
    
    
    def get_cos(self, angle):
        # angle ABC
        if isinstance(angle, tuple):
            angle = ''.join(angle)
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
        if isinstance(angle, tuple):
            angle = ''.join(angle)
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
    
    def get_cross_product(self, line_1, line_2):
        # AB \times CD
        p1, p2 = line_1
        p3, p4 = line_2
        xa, ya = self.p_pos[p1]
        xb, yb = self.p_pos[p2]
        xc, yc = self.p_pos[p3]
        xd, yd = self.p_pos[p4]
        AB = (xb - xa, yb - ya)
        CD = (xd - xc, yd - yc)
        return AB[0] * CD[1] - AB[1] * CD[0]
    
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
            if p not in self.p_pos:
                self.p_pos[p] = None
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
    
    def update_symbol(self, solution, char, constraint=[], cannot_coincide=True):
        # select one solution and constraint, save to global and update p_pos
        syms = get_symbol(char)
        # update p_pos with new solved symbols
        if len(solution) == 0:
            return None
        if type(solution) == dict:
            x = solution.get(syms[0], self.p_pos[char][0])
            y = solution.get(syms[1], self.p_pos[char][1])
            if x.has(I) or y.has(I):
                x = simplify_and_trim(x)
                y = simplify_and_trim(y)
            if len(constraint) != 0:
                self.p_pos_range[char] = constraint[0]
            if type(x) in [Float, float] and type(y) in [Float, float]:
                coincide_ps = self.get_coincide_points(x, y)
                if cannot_coincide and len(coincide_ps) > 0:
                    return None
                
        elif type(solution) == list:
            # have multiple solutions, choose reasonable 
            # 1. solution like [(x_expr, y_expr), (x_expr, y_expr)]
            # 2 .solution like [(1, 2), (3, 4)]
            solution_ = []
            constraint_ = []
            coincide_flags = []
            for s in solution:
                # select correspond constraint
                # 1. len constrants = 2, len self.p_pos_range = 0
                # 2. len constrants = 0, len self.p_pos_range = 1
                # 3. len constrants = 0, len self.p_pos_range = 0
                
                if len(constraint) != 0:
                    constraint_i = constraint[solution.index(s)]
                elif char in self.p_pos_range :
                    constraint_i = self.p_pos_range.get(char)
                else:
                    constraint_i = None
                # if constriant_i solved to be false
                if constraint_i == False:
                    continue
                # if imaginary part have a very small coefficient (<1e-5)
                if s[0].has(I) or s[1].has(I):
                    s = (simplify_and_trim(s[0]), simplify_and_trim(s[1]))
                # if still have imaginary part
                if s[0].has(I) or s[1].has(I):
                    continue
                # when solution is float value:
                if type(s[0]) in [Float, float] and type(s[1]) in [Float, float]:
                    # don't have big numbers
                    if abs(s[0]) > 1e3 or abs(s[1]) > 1e3:
                        continue
                    # must satisfy constraint_i
                    if constraint_i != None:
                        if not constraint_i.subs({syms[0]: s[0], syms[1]: s[1]}):
                            continue
                    # record whether coincide
                    coincide = len(self.get_coincide_points(s[0], s[1])) > 0
                    coincide_flags.append(coincide)
                
                # when solution is expression:
                else: # delete solution with too small interval
                    if constraint_i != None:
                        if type(constraint_i) == Or:
                            continue
                        sym = list(constraint_i.free_symbols)[0]
                        if constraint_i.args[0].lhs.has(sym):
                            bound_1 = float(constraint_i.args[0].rhs)
                        else:
                            bound_1 = float(constraint_i.args[0].lhs)
                        if constraint_i.args[1].lhs.has(sym):
                            bound_2 = float(constraint_i.args[1].rhs)
                        else:
                            bound_2 = float(constraint_i.args[1].lhs)
                        if abs(bound_1 - bound_2) < 0.1:
                            continue
                if constraint_i != None:
                    constraint_.append(constraint_i)
                solution_.append(s)
            
            # try to remove coincide solution
            # choose solution not coincide, if both coincide: 
            # if cannot_coincide: return None
            # else random choose
            if True in coincide_flags:
                if False in coincide_flags:
                    solution_ = [solution_[coincide_flags.index(False)]]
                else:
                    if cannot_coincide:
                        solution_ = []
                    else:
                        pass
                        
            if len(solution_) == 0:
                return None
            
            x, y = random.choice(solution_)
            # add constraints for symbols of p
            if len(constraint_) != 0:
                self.p_pos_range[char] = constraint_[solution_.index((x, y))]
                
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
        return self.p_pos[char]
            
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
        # 4. future point position depend on this point
        for p in self.points:
            if len(self.clause_subset[p]) == 0:
                if self.p_pos[p] is not None:
                    x, y = self.p_pos[p]
                    if isinstance(x, Expr) or isinstance(y, Expr):
                        depend_flag = self.check_dependency(p)
                        if depend_flag:
                            position = self.random_allocate_position(target_p=p)
                            self.update_values(p, position)
    
    def check_dependency(self, target_p):
        for p in self.points:
            if p <= target_p:
                continue
            clause_subset = self.clause_subset[p]
            items = [parse_clause(c)[1] for c in clause_subset]
            items = [item for sublist in items for item in sublist]
            for item in items:
                if target_p in item:
                    return True
            
        return False
    
    def random_substitude_value(self, x_expr, y_expr, interval=None):
        syms = set(list(x_expr.free_symbols) + list(y_expr.free_symbols))
        syms = list(syms)
        if len(syms) == 1:
            sym = syms[0]
            value = random.uniform(interval[0], interval[1])
            x_val, y_val = x_expr.subs(sym, value), y_expr.subs(sym, value)
            if max(abs(x_val), abs(y_val)) > 25:
                x_val = x_expr.subs(sym, value / max(x_val, y_val) * 10)
                y_val = y_expr.subs(sym, value / max(x_val, y_val) * 10)
                
        elif len(syms) == 2:
            x_val = random.uniform(interval[0], interval[1])
            y_val = random.uniform(interval[0], interval[1])
            
        return x_val, y_val
    
    def random_allocate_position(self, target_p=None, n=5):
        # random allocate for n times
        # choose the pos that \max \min dist(p, p_i), p_i in self.p_pos
        
        # determine the interval of random numbers first
        interval_of_random = []
        # target_p is None
        if target_p == None or self.p_pos[target_p] == None:
            interval_of_random = [(5, 10) for i in range(n)]
        
        # position of target_p is float value
        elif type(self.p_pos[target_p][0]) in [float, Float, int] and \
            type(self.p_pos[target_p][1]) in [float, Float, int]:
            return self.p_pos[target_p]
        
        else: # position of target_p is expression
            constraint = self.p_pos_range.get(target_p, None)

            # has constraint, get upper and lower bounds and devide
            x_expr, y_expr = self.p_pos[target_p]
            syms = list(set(list(x_expr.free_symbols) + 
                            list(y_expr.free_symbols)))
            if len(syms) == 1:
                sym = syms[0]
                
                if constraint is None or type(constraint) == Or:
                    constraint = solve((-20 < x_expr, x_expr < 20, -20 < y_expr, y_expr < 20), sym)
                
                if constraint == False:
                    interval_of_random = [(5, 10) for i in range(n)]
                else:
                    # transfer ineq to (..) And (..)
                    if type(constraint) == StrictLessThan:
                        constraint = (-20 < sym) & constraint
                    if type(constraint) == StrictGreaterThan:
                        constraint = constraint & (sym < 20)
                    try:
                        if constraint.args[0].lhs.has(sym):
                            bound_1 = float(constraint.args[0].rhs)
                        else:
                            bound_1 = float(constraint.args[0].lhs)
                        if constraint.args[1].lhs.has(sym):
                            bound_2 = float(constraint.args[1].rhs)
                        else:
                            bound_2 = float(constraint.args[1].lhs)
                    except Exception as e:
                        print(e)
                    inf = max([min([bound_1, bound_2]), -10])
                    sup = min([max([bound_1, bound_2]), 10])
                    val = (sup - inf) / n
                    interval_of_random = [(inf + i*val, inf + (i+1)*val) for i in range(n)]

            else:
                interval_of_random = [(5, 10) for i in range(n)]
        
        max_distance = 0
        best_point = None
        for interval in interval_of_random:
            if target_p == None or self.p_pos[target_p] == None:
                # x: (-10, 10), y: (5, 10)
                x = random.uniform(- interval[1], interval[1])
                y = random.uniform(interval[0], interval[1])
            else:
                x, y = self.p_pos[target_p]
                x, y = self.random_substitude_value(x, y, interval=interval)
            
            if im(x).simplify() != 0 or im(y).simplify() != 0:
                continue
                
            min_distance = float('inf')

            for p, pos in self.p_pos.items():
                if pos is None or type(pos[0]) not in [float, Float, int]:
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
            n_more_lines=2
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
        
        allocator = Allocator(cg.states, c_cdls, t_cdls)
        allocator.allocate()
        
        print("---------- Location ----------")
        for p, pos in allocator.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
    
        print('==============================================')

        