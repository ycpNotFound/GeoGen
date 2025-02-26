import copy
import itertools
import math
import time
import warnings
from copy import deepcopy
from itertools import combinations

from sympy import symbols

from formalgeo.core import EquationKiller as EqKiller
from formalgeo.parse import get_equation_from_tree, parse_expr
from formalgeo.problem.condition import Condition, Goal
from formalgeo.tools import rough_equal


class Problem:
    def __init__(self, p_pos=None):
        """Problem conditions, goal, and solving message."""
        self.parsed_predicate_GDL = None
        self.parsed_theorem_GDL = None
        self.parsed_problem_CDL = None
        self.condition = None  # <Condition>, all conditions of current problem.
        self.goal = None  # <Goal>, problem goal.
        self.timing = {}  # <dict>, {step: (theorem, timing)}, such as {0: ('init_problem', 0.00325)}.
        self.p_pos = p_pos
        
    def sort_counter_clocksiwe_angle(self, points):
        if self.p_pos is None:
            return points
        (A, B, C) = [self.p_pos[p] for p in points]
        AB = (B[0] - A[0], B[1] - A[1])
        BC = (C[0] - B[0], C[1] - B[1])
        cross_product_z = AB[0]*BC[1] - AB[1]*BC[0]
        if cross_product_z > 0:
            return tuple([points[2], points[1], points[0]])
        else:
            return points
        
    def sort_counter_clockwise_cocircular(self, centre, points):
        if self.p_pos is None:
            return points
        O = self.p_pos[centre]

        def angle_from_origin(point):
            dx = point[0] - O[0]
            dy = point[1] - O[1]
            return math.atan2(dy, dx)
        angles = [(key, angle_from_origin(self.p_pos[key])) for key in points]
        sorted_angles = sorted(angles, key=lambda x: x[1], reverse=True)
        sorted_points = [item[0] for item in sorted_angles]
        start_idx = sorted_points.index(min(points))
        sorted_points = sorted_points[start_idx:] + sorted_points[:start_idx]
        return tuple(sorted_points)
    
    def sort_by_x_collinear(self, points):
        if self.p_pos is None:
            return points
        sorted_points = sorted(points, key=lambda x: self.p_pos[x][0])
        return tuple(sorted_points)

    def load_problem_by_fl(self, parsed_predicate_GDL, parsed_theorem_GDL, parsed_problem_CDL):
        """Load problem through problem CDL."""
        self.parsed_predicate_GDL = parsed_predicate_GDL  # gdl
        self.parsed_theorem_GDL = parsed_theorem_GDL  # gdl
        self.parsed_problem_CDL = parsed_problem_CDL  # cdl
        fix_length_predicates = list(self.parsed_predicate_GDL["Preset"]["FixLength"])
        fix_length_predicates += list(self.parsed_predicate_GDL["Entity"])
        fix_length_predicates += list(self.parsed_predicate_GDL["Relation"])
        variable_length_predicates = list(self.parsed_predicate_GDL["Preset"]["VariableLength"])
        self.condition = Condition()
        self.condition.init_by_fl(fix_length_predicates, variable_length_predicates)

        self._construction_init()  # start construction

        # conditions of text_and_image
        for predicate, item in self.parsed_problem_CDL["parsed_cdl"]["text_and_image_cdl"]:
            if predicate == "Equal":
                self.add("Equation", get_equation_from_tree(self, item),
                         (-1,), ("prerequisite", None, None))
            elif predicate == "Equation":
                self.add("Equation", parse_expr(self, item),
                         (-1,), ("prerequisite", None, None))
            elif predicate == 'ParallelBetweenLine' and self.p_pos is not None:
                item_1, item_2 = self.same_side_for_parallel(item)
                self.add(predicate, tuple(item_1), (-1,), ("prerequisite", None, None))
                self.add(predicate, tuple(item_2), (-1,), ("prerequisite", None, None))
            elif predicate == 'IsBisectorOfAngle':
                item = item[:2] + list(self.sort_counter_clocksiwe_angle(item[2:]))
                self.add(predicate, tuple(item), (-1,), ("prerequisite", None, None))
            else:
                self.add(predicate, tuple(item), (-1,), ("prerequisite", None, None))

        self.goal = Goal()  # set goal
        self.goal.init_by_fl(self, self.parsed_problem_CDL["parsed_cdl"]["goal"])

    def load_problem_by_copy(self, problem):
        """Load problem through copying existing problem."""
        self.parsed_predicate_GDL = problem.parsed_predicate_GDL  # gdl
        self.parsed_theorem_GDL = problem.parsed_theorem_GDL  # gdl
        self.parsed_problem_CDL = problem.parsed_problem_CDL  # cdl
        self.condition = Condition()  # copy all msg of problem
        self.condition.init_by_copy(problem.condition)
        self.timing = copy.deepcopy(problem.timing)
        self.goal = Goal()  # set goal
        self.goal.init_by_copy(problem.goal)

    def same_side_for_parallel(self, item):
        # AB // CD, AC in one side, BD in the other side
        A, B, C, D = item
        AB = [self.p_pos[B][0] - self.p_pos[A][0],
                  self.p_pos[B][1] - self.p_pos[A][1]]
        CD = [self.p_pos[D][0] - self.p_pos[C][0],
                  self.p_pos[D][1] - self.p_pos[C][1]]
        if AB[0] * CD[0] + AB[1] * CD[1] > 0:
            return (A, B, C, D), (C, D, A, B)
        else:
            return (A, B, D, C), (D, C, A, B)
    
    def _construction_init(self):
        """
        Constructive process.
        1.Collinear expand.
        2.Cocircular expand.
        3.Shape expand. Shape(s1,s2,s3), Shape(s3,s2,s4) ==> Shape(s1,s4).
        4.Angle expand (combination).
        5.Angle expand (collinear).
        6.Angle expand (find possible angle).
        """
        # 0. Add all point and line instances.
        for line in self.parsed_problem_CDL['lines']:
            for i in range(len(line)-1):
                line_unit = (line[i], line[i+1])
                line_unit_rev = (line[i+1], line[i])
                self.condition.add('Line', line_unit, (-1,), ("prerequisite", None, None))
                self.condition.add('Line', line_unit_rev, (-1,), ("prerequisite", None, None))
        for point in self.parsed_problem_CDL['points']:
            self.condition.add('Point', (point, ), (-1, ), ("prerequisite", None, None)) 
            
                    
        # 1.Collinear expand.
        for predicate, item in self.parsed_problem_CDL["parsed_cdl"]["construction_cdl"]:  # Collinear
            if predicate != "Collinear":
                continue
            if not self.fv_check("Collinear", item):  # FV check
                w_msg = "FV check not passed: [{}, {}]".format(predicate, item)
                warnings.warn(w_msg)
                continue

            added, _id = self.condition.add(predicate, tuple(item), (-1,), ("prerequisite", None, None))
            if not added:
                continue

            self.condition.add(predicate, tuple(item[::-1]), (_id,), ("extended", None, None))
            self.add("Line", (item[0], item[-1]), (_id,), ("extended", None, None))
            for extended_item in combinations(item, 3):  # l=3 is enough
                self.condition.add("Collinear", extended_item, (_id,), ("extended", None, None))
                self.condition.add("Collinear", extended_item[::-1], (_id,), ("extended", None, None))
                self.add("Angle", extended_item, (_id,), ("extended", None, None))
                self.add("Angle", extended_item[::-1], (_id,), ("extended", None, None))
                angle_sym_1 = self.get_sym_of_attr('MeasureOfAngle', extended_item)
                angle_sym_2 = self.get_sym_of_attr('MeasureOfAngle', extended_item[::-1])
                self.condition.add("Equation", angle_sym_1 - 180, (_id, ), ("extended", None, None))
                self.condition.add("Equation", angle_sym_2 - 180, (_id, ), ("extended", None, None))

        # 2.Cocircular expand.
        for predicate, item in self.parsed_problem_CDL["parsed_cdl"]["construction_cdl"]:  # Cocircular
            if predicate != "Cocircular":
                continue
            if not self.fv_check("Cocircular", item):  # FV check
                w_msg = "FV check not passed: [{}, {}]".format(predicate, item)
                warnings.warn(w_msg)
                continue

            added, _id = self.condition.add(predicate, tuple(item), (-1,), ("prerequisite", None, None))
            if not added:
                continue

            circle = item[0]
            self.add("Circle", (circle,), (_id,), ("extended", None, None))
            if len(item) == 1:
                continue

            item = item[1:]
            for com in range(1, len(item) + 1):  # extend cocircular
                for extended_item in combinations(item, com):
                    if com == 2:
                        self.condition.add("Arc", (circle, extended_item[0], extended_item[-1]),
                                           (_id,), ("extended", None, None))
                        self.condition.add("Arc", (circle, extended_item[-1], extended_item[0]),
                                           (_id,), ("extended", None, None))
                    cocircular = list(extended_item)
                    l = len(cocircular)
                    for bias in range(l):
                        extended_item = tuple([circle] + [cocircular[(i + bias) % l] for i in range(l)])
                        self.condition.add("Cocircular", extended_item, (_id,), ("extended", None, None))

        # 3.Shape expand.
        jigsaw_unit = {}  # shape's jigsaw
        shape_unit = []  # mini shape unit
        for predicate, item in self.parsed_problem_CDL["parsed_cdl"]["construction_cdl"]:  # Shape
            if predicate != "Shape":
                continue
            if not self.fv_check("Shape", item):  # FV check
                w_msg = "FV check not passed: [{}, {}]".format(predicate, item)
                warnings.warn(w_msg)
                continue

            if len(item) == 1:  # point or line
                if len(item[0]) == 1:
                    self.add("Point", tuple(item[0]), (-1,), ("prerequisite", None, None))
                else:
                    self.add("Line", tuple(item[0]), (-1,), ("prerequisite", None, None))
                continue
            elif len(item) == 2 and len(item[0]) == 2 and len(item[1]) == 2:  # angle
                self.add("Angle", tuple(item[0] + item[1][1]), (-1,), ("prerequisite", None, None))
                continue

            added, all_forms = self._add_shape(tuple(item), (-1,), ("prerequisite", None, None))  # shape
            if not added:
                continue

            for shape in all_forms:
                jigsaw_unit[shape] = all_forms
                shape_unit.append(shape)

        shape_comb = shape_unit
        jigsaw_comb = jigsaw_unit
        while len(shape_comb):
            shape_comb_new = []
            jigsaw_comb_new = {}
            for unit in shape_unit:
                for comb in shape_comb:

                    if len(unit[-1]) != len(comb[0]):  # has same sides?
                        continue
                    elif len(unit[-1]) == 3:  # is arc and same?
                        if unit[-1] != comb[0]:
                            continue
                    else:
                        if unit[-1] != comb[0][::-1]:  # is line and same?
                            continue

                    if unit in jigsaw_comb[comb]:  # comb is combined from unit
                        continue

                    same_length = 1  # number of same sides
                    mini_length = len(unit) if len(unit) < len(comb) else len(comb)  # mini length
                    while same_length < mini_length:
                        if len(unit[- same_length - 1]) != len(comb[same_length]):  # all arcs or all lines
                            break
                        elif len(unit[- same_length - 1]) == 3:  # arc
                            if unit[- same_length - 1] != comb[same_length]:
                                break
                        else:  # line
                            if unit[- same_length - 1] != comb[same_length][::-1]:
                                break

                        same_length += 1

                    new_shape = list(unit[0:len(unit) - same_length])  # diff sides in polygon1
                    new_shape += list(comb[same_length:len(comb)])  # diff sides in polygon2

                    if not len(new_shape) == len(set(new_shape)):  # ensure no ring
                        continue

                    new_shape = tuple(new_shape)
                    if self.condition.has("Shape", new_shape):
                        continue

                    all_sides = ""
                    for item in new_shape:  # remove circle center point
                        if len(item) == 3:
                            item = item[1:]
                        all_sides += item
                    checked = True
                    for point in all_sides:
                        if all_sides.count(point) > 2:
                            checked = False
                            break
                    if not checked:  # ensure no holes
                        continue

                    premise = (self.condition.get_id_by_predicate_and_item("Shape", unit),
                               self.condition.get_id_by_predicate_and_item("Shape", comb))

                    added, all_forms = self._add_shape(new_shape, premise, ("extended", None, None))  # add shape
                    if not added:  # ensure added
                        continue

                    new_shape_jigsaw = jigsaw_unit[unit] | jigsaw_comb[comb]
                    for shape in all_forms:
                        jigsaw_comb_new[shape] = new_shape_jigsaw
                        shape_comb_new.append(shape)

            shape_comb = shape_comb_new
            jigsaw_comb = jigsaw_comb_new

        # 4.Angle expand (combination).
        angle_unit = self.condition.get_items_by_predicate("Angle")
        jigsaw_unit = {}
        for angle in angle_unit:
            jigsaw_unit[angle] = {angle}

        angle_comb = angle_unit  # combination angle
        jigsaw_comb = jigsaw_unit  # angle's jigsaw
        while len(angle_comb):
            angle_comb_new = []
            jigsaw_comb_new = {}
            for unit in angle_unit:
                for comb in angle_comb:

                    if unit in jigsaw_comb[comb]:  # comb is combined from unit
                        continue

                    if not (unit[1] == comb[1] and unit[2] == comb[0] and unit[0] != comb[2]):  # ensure adjacent
                        continue

                    angles = self.condition.get_items_by_predicate("Angle")
                    if (unit[0], unit[1], comb[2]) in angles or \
                            (unit[0], comb[2], unit[1]) in angles or \
                            (comb[2], unit[0], unit[1]) in angles:
                        continue

                    new_angle = (unit[0], unit[1], comb[2])
                    new_angle = self.sort_counter_clocksiwe_angle(new_angle)

                    if not len(new_angle) == len(set(new_angle)):  # ensure same points
                        continue

                    premise = (self.condition.get_id_by_predicate_and_item("Angle", unit),
                               self.condition.get_id_by_predicate_and_item("Angle", comb))
                    added, _ = self.condition.add("Angle", new_angle, premise,
                                                  ("extended", None, None))  # need to expand line
                    if not added:
                        continue

                    new_angle_jigsaw = jigsaw_unit[unit] | jigsaw_comb[comb]
                    jigsaw_comb_new[new_angle] = new_angle_jigsaw
                    angle_comb_new.append(new_angle)

            angle_comb = angle_comb_new
            jigsaw_comb = jigsaw_comb_new

        # 5.Angle collinear expand.
        for angle in self.condition.get_items_by_predicate("Angle"):
            premise = (self.condition.get_id_by_predicate_and_item("Angle", angle),)
            for same_angle in self._get_same_angles(angle):
                self.add("Angle", same_angle, premise, ("extended", None, None))

        # 6.Angle expand (vertical angle).
        # for angle in self.condition.get_items_by_predicate("Angle"):
        #     premise = (self.condition.get_id_by_predicate_and_item("Angle", angle),)
        #     self.add("Angle", (angle[2], angle[1], angle[0]), premise, ("extended", None, None))

        # 6.Angle expand (find possible angle).
        def collinear(p1, p2, p3):
            collinear_ls = self.condition.get_items_by_predicate("Collinear")
            for l in collinear_ls:
                if len(set(l) & set((p1, p2, p3))) == 3:
                    return True
            if self.p_pos is not None:
                pos_1 = self.p_pos[p1]
                pos_2 = self.p_pos[p2]
                pos_3 = self.p_pos[p3]
                AB = (pos_2[0] - pos_1[0], pos_2[1] - pos_1[1])
                AC = (pos_3[0] - pos_1[0], pos_3[1] - pos_1[1])
                cross_product = AB[0] * AC[1] - AB[1] * AC[0]
                if abs(cross_product) < 1e-5:
                    return True
            return False
        
        for line in self.condition.get_items_by_predicate("Line"):
            # AB is line, find other lines starts with B (BC, BD, ...)
            other_lines = [
                l for l in self.condition.get_items_by_predicate("Line")
                if len(set(l) & set(line)) != 2 and l[0] == line[1]
                and not collinear(line[0], l[0], l[1])
            ]
            for l in other_lines:
                new_angle = (line[0], l[0], l[1])
                premise = (self.condition.get_id_by_predicate_and_item("Line", line),
                           self.condition.get_id_by_predicate_and_item("Line", l))
                self.add("Angle", new_angle, premise, ("extended", None, None))

        # 6.Cocircular radius equal (new added).
        # for predicate, item in self.parsed_problem_CDL["parsed_cdl"]["construction_cdl"]:  # Cocircular
        #     if predicate != "Cocircular":
        #         continue
        #     if not self.fv_check("Cocircular", item):  # FV check
        #         w_msg = "FV check not passed: [{}, {}]".format(predicate, item)
        #         warnings.warn(w_msg)
        #         continue
        #     radius = []
        #     for p in item[1:]:
        #         line = (item[0], p)
        #         if line in self.condition.items_group['Line']:
        #             radius.append(line)
            
        #     for i in range(len(radius)-1):
        #         l1, l2 = radius[i], radius[i+1]
        #         l1_sym = self.get_sym_of_attr('LengthOfLine', l1)
        #         l2_sym = self.get_sym_of_attr('LengthOfLine', l2)
        #         premise = [
        #             self.condition.get_id_by_predicate_and_item(predicate, tuple(item)),
        #             self.condition.get_id_by_predicate_and_item("Line", l1),
        #             self.condition.get_id_by_predicate_and_item("Line", l2),
        #         ]
        #         self.add("Equation", l1_sym - l2_sym, premise, ("extended", None, None))
        return 


    def _add_shape(self, shape, premise, theorem):
        """pass"""
        added, _id = self.condition.add("Shape", shape, premise, theorem)
        if not added:
            return False, None

        all_forms = [shape]
        l = len(shape)
        for bias in range(1, l):  # all forms
            new_item = tuple([shape[(i + bias) % l] for i in range(l)])
            self.condition.add("Shape", new_item, (_id,), ("extended", None, None))
            all_forms.append(new_item)

        shape = list(shape)
        _, col = self.condition.get_ids_and_items_by_predicate_and_variable("Collinear", ["a", "b", "c"])
        i = 0
        has_arc = False
        while i < len(shape):
            if len(shape[i]) == 2:
                self.add("Line", (shape[i][0], shape[i][1]),
                         (_id,), ("extended", None, None))
            else:
                has_arc = True
                i += 1
                continue

            j = (i + 1) % len(shape)
            if len(shape[j]) == 2:
                self.add("Angle", (shape[i][0], shape[i][1], shape[j][1]),
                         (_id,), ("extended", None, None))  # extend angle
                if (shape[i][0], shape[i][1], shape[j][1]) in col:
                    shape[i] = shape[i][0] + shape[j][1]
                    shape.pop(j)
                    continue
            i += 1

        if not has_arc and len(shape) > 2:  # extend polygon
            valid = True
            i = 0
            l = len(shape)
            while valid and i < l:
                if shape[i][1] != shape[(i + 1) % l][0]:
                    valid = False
                i += 1
            if valid:
                self.add("Polygon", tuple([item[0] for item in shape]),
                         (_id,), ("extended", None, None))

        return True, set(all_forms)

    def _get_same_angles(self, angle):
        """
        Expanding angles according to collinear.
        Angle(ABC), Collinear(BCD)  ==>  Angle(ABD)
        """
        collinear = []
        for _id in self.condition.ids_of_predicate["Collinear"]:
            if self.condition.items[_id][3] == ("prerequisite", None, None):
                collinear.append(self.condition.items[_id][1])

        a, v, b = angle
        a_collinear = None
        b_collinear = None
        for collinear_points in collinear:
            if v in collinear_points:
                if a in collinear_points:
                    a_collinear = collinear_points
                if b in collinear_points:
                    b_collinear = collinear_points

        a_points = []  # Points collinear with a and on the same side with a
        b_points = []
        if a_collinear is not None:
            if a_collinear.index(v) < a_collinear.index(a):  # .....V...P..
                i = a_collinear.index(v) + 1
                while i < len(a_collinear):
                    a_points.append(a_collinear[i])
                    i += 1
            else:  # ...P.....V...
                i = 0
                while i < a_collinear.index(v):
                    a_points.append(a_collinear[i])
                    i += 1
        else:
            a_points.append(a)

        if b_collinear is not None:
            if b_collinear.index(v) < b_collinear.index(b):  # .....V...P..
                i = b_collinear.index(v) + 1
                while i < len(b_collinear):
                    b_points.append(b_collinear[i])
                    i += 1
            else:  # ...P.....V...
                i = 0
                while i < b_collinear.index(v):
                    b_points.append(b_collinear[i])
                    i += 1
        else:
            b_points.append(b)

        same_angles = []  # Same angle get by collinear
        for a_point in a_points:
            for b_point in b_points:
                same_angles.append((a_point, v, b_point))
                # reverse angle regards to be same
                # same_angles.append((b_point, v, a_point))

        return same_angles

    def add_collinear_extend(self, item, _id):
        item_rev = tuple(list(item)[::-1])
        self.condition.add('Collinear', item_rev, (_id, ), ("extended", None, None))
        # add sub lines
        for i in range(len(item)-1):
            new_line = (item[i], item[i+1])
            new_line_rev = (item[i+1], item[i])
            self.condition.add('Line', new_line, (_id, ), ("extended", None, None))
            self.condition.add('Line', new_line_rev, (_id, ), ("extended", None, None))
        # add flat angles
        for extended_item in combinations(item, 3):  # l=3 is enough
            self.condition.add("Collinear", extended_item, (_id,), ("extended", None, None))
            self.condition.add("Collinear", extended_item[::-1], (_id,), ("extended", None, None))
            self.condition.add("Angle", extended_item, (_id,), ("extended", None, None))
            self.condition.add("Angle", extended_item[::-1], (_id,), ("extended", None, None))
            angle_sym_1 = self.get_sym_of_attr('MeasureOfAngle', extended_item)
            angle_sym_2 = self.get_sym_of_attr('MeasureOfAngle', extended_item[::-1])
            self.condition.add("Equation", angle_sym_1 - 180, (_id, ), ("extended", None, None))
            self.condition.add("Equation", angle_sym_2 - 180, (_id, ), ("extended", None, None))

        # add potential collinear
        for ps in self.condition.get_items_by_predicate('Collinear'):
            if len(set(item) & set(ps)) == 2:
                ps_co = tuple(set(item) & set(ps))
                p1 = tuple(set(item) - set(ps_co))[0]
                p2 = tuple(set(ps) - set(ps_co))[0]
                for p in ps_co:
                    new_item = self.sort_by_x_collinear((p1, p, p2))
                    new_item_rev = tuple(list(new_item)[::-1])
                    _pre = self.condition.get_id_by_predicate_and_item('Collinear', ps)
                    self.condition.add('Collinear', new_item, (_id, _pre), ("extended", None, None))
                    self.condition.add('Collinear', new_item_rev, (_id, _pre), ("extended", None, None))
        
        def rotations(t):
            return [t[i:] + t[:i] for i in range(len(t))]
        
        # add potential shape 
        # A M B collinear, find lines like with XM
        lines_XM = [l for l in self.condition.get_items_by_predicate('Line') if item[1] == l[1]]
        # find lines like MY
        lines_MY = [l for l in self.condition.get_items_by_predicate('Line') if item[1] == l[0]]
        # check whether shape XMY exists
        for l1, l2 in itertools.product(lines_XM, lines_MY):
            if len(set(l1) & set(l2)) == 2:
                continue
            if len(set(l1).union(l2) & set(item)) == 3:
                continue
            # check whether points of l1, l2 collinear
            collinear_flag = False
            for collinear_ps in self.condition.get_items_by_predicate('Collinear'):
                if len(set(collinear_ps) & set([l1[0], l1[1], l2[1]])) == 3:
                    collinear_flag = True
                    break
            if collinear_flag:
                continue

            l3 = (l1[0], l2[1])
            if l3 in self.condition.get_items_by_predicate('Line'):
                poly = (l1[0], l1[1], l2[1])
                if poly in self.condition.get_items_by_predicate('Collinear'):
                    continue
                poly = self.sort_counter_clocksiwe_angle(poly)
                self.add('Polygon', poly, (_id, ), ("extended", None, None))
                for angle in rotations(poly):
                    self.add("Angle", angle, (_id, ), ("extended", None, None))
                
    def add_cocircular_extend(self, item, _id):
        ori_ps = self.condition.items_group['Cocircular'][0]
        new_ps = [p for p in item if p not in ori_ps]
        for p in new_ps:
            self.condition.add('Cocircular', (ori_ps[0], p), (_id, ), ("extended", None, None))
            for ori_ps_i in deepcopy(self.condition.items_group['Cocircular']):
                c, ps_on_c = ori_ps_i[0], ori_ps_i[1:]
                added_ps = tuple(set(ps_on_c + (p, )))
                added_ps = self.sort_counter_clockwise_cocircular(c, added_ps)
                added_ps = (c, ) + added_ps
                self.condition.add('Cocircular', added_ps, (_id, ), ("extended", None, None))
        
    def add(self, predicate, item, premise, theorem, skip_check=False):
        """
        Add item to condition of specific predicate category.
        Also consider condition expansion and equation construction.
        :param predicate: Construction, Entity, Relation or Equation.
        :param item: <tuple> or equation.
        :param premise: tuple of <int>, premise of item.
        :param theorem: <tuple>, (t_name, t_branch, t_para).
        :param skip_check: <bool>, set to True when you are confident that the format of item must be legal.
        :return: True or False.
        """
        if not skip_check and not self.check(predicate, item, premise, theorem):
            return False
        # special cases:
        if predicate == 'ParallelBetweenLine':
            for collinear_ps in self.condition.get_items_by_predicate('Collinear'):
                if len(set(collinear_ps) & set([item[0], item[1], item[2]])) == 3:
                    return False
                if len(set(collinear_ps) & set([item[0], item[1], item[3]])) == 3:
                    return False
        elif predicate == 'Collinear':
            item = self.sort_by_x_collinear(item)
        elif predicate == 'Angle':
            if self.p_pos is not None:
                # check the points in angle are counter clockwise
                # self.p_pos: {'p': [x, y]}
                p1, p2, p3 = item
                p1_pos = self.p_pos[p1]
                p2_pos = self.p_pos[p2]
                p3_pos = self.p_pos[p3]
                # Calculate the vectors
                v1 = (p2_pos[0] - p1_pos[0], p2_pos[1] - p1_pos[1])
                v2 = (p3_pos[0] - p1_pos[0], p3_pos[1] - p1_pos[1])
                # Calculate the cross product: p2-p1 x p3-p1 > 0 (in cv2 < 0) is ccw
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross > 1e-5: # if cross > 0, is not ccw
                    return False
                

        added, _id = self.condition.add(predicate, item, premise, theorem)
        if added:
            if predicate == "Equation":  # preset Equation
                return True

            if predicate in self.parsed_predicate_GDL["Preset"]["BasicEntity"]:  # preset BasicEntity
                if predicate == "Line":
                    self.condition.add("Line", item[::-1], (_id,), ("extended", None, None))
                    self.condition.add("Point", (item[0],), (_id,), ("extended", None, None))
                    self.condition.add("Point", (item[1],), (_id,), ("extended", None, None))
                elif predicate == "Arc":
                    self.condition.add("Point", (item[1],), (_id,), ("extended", None, None))
                    self.condition.add("Point", (item[2],), (_id,), ("extended", None, None))
                elif predicate == "Angle":
                    self.add("Line", (item[0], item[1]),
                             (_id,), ("extended", None, None), skip_check=True)
                    self.add("Line", (item[1], item[2]),
                             (_id,), ("extended", None, None), skip_check=True)
                    # define symbol for angle
                    sym = self.get_sym_of_attr('MeasureOfAngle', item)
                    a = 1
                elif predicate == "Polygon":
                    l = len(item)
                    for bias in range(1, l):  # all forms
                        new_item = tuple([item[(i + bias) % l] for i in range(l)])
                        self.condition.add("Polygon", new_item, (_id,), ("extended", None, None))
                return True  # Point and Circle no need to extend

            if predicate in self.parsed_predicate_GDL["Entity"]:  # user defined Entity
                item_GDL = self.parsed_predicate_GDL["Entity"][predicate]
            elif predicate in self.parsed_predicate_GDL["Relation"]:  # user defined Relation
                item_GDL = self.parsed_predicate_GDL["Relation"][predicate]
                
            else: # other cases
                if predicate == 'Collinear':
                    self.add_collinear_extend(item, _id)
                elif predicate == 'Cocircular':
                    self.add_cocircular_extend(item, _id)
                else:
                    pass
                return True

            predicate_vars = item_GDL["vars"]
            letters = {}  # used for vars-letters replacement
            for i in range(len(predicate_vars)):
                letters[predicate_vars[i]] = item[i]

            for para_list in item_GDL["multi"]:  # multi
                self.condition.add(predicate, tuple(letters[i] for i in para_list), (_id,), ("extended", None, None))

            for extended_predicate, para in item_GDL["extend"]:  # extended
                if extended_predicate == "Equal":
                    self.add("Equation", get_equation_from_tree(self, para, True, letters),
                             (_id,), ("extended", None, None))
                elif extended_predicate == 'PerpendicularBetweenLine':
                    # check counter clockwise (ccw)
                    points = tuple(letters[i] for i in para)[:3]
                    points_ccw = self.sort_counter_clocksiwe_angle(points)
                    points_input = points_ccw + (points[1], )
                    self.add(extended_predicate, points_input,
                             (_id,), ("extended", None, None))
                else:
                    self.add(extended_predicate, tuple(letters[i] for i in para),
                             (_id,), ("extended", None, None))

            return True

        return False

    def check(self, predicate, item, premise=None, theorem=None):
        """
        EE check and FV check.
        :param predicate: Construction, Entity, Relation or Equation.
        :param item: <tuple> or equation.
        :param premise: tuple of <int>, premise of item.
        :param theorem: <str>, theorem of item.
        :return: True or False.
        """
        if predicate not in self.condition.items_group:  # predicate must be defined
            e_msg = "Predicate '{}' not defined in current predicate GDL.".format(predicate)
            raise Exception(e_msg)
        if not self.ee_check(predicate, item):  # ee check
            w_msg = "EE check not passed: [{}, {}, {}, {}]".format(predicate, item, premise, theorem)
            # warnings.warn(w_msg)
            return False
        if not self.fv_check(predicate, item):  # fv check
            w_msg = "FV check not passed: [{}, {}, {}, {}]".format(predicate, item, premise, theorem)
            # warnings.warn(w_msg)
            return False

        return True

    def ee_check(self, predicate, item):
        """Entity Existence check."""

        if predicate == "Equation" or predicate in self.parsed_predicate_GDL["Preset"]["BasicEntity"] \
                or predicate in self.parsed_predicate_GDL["Preset"]["Construction"]:
            return True
        elif predicate in self.parsed_predicate_GDL["Entity"]:
            item_GDL = self.parsed_predicate_GDL["Entity"][predicate]
        elif predicate in self.parsed_predicate_GDL["Relation"]:
            item_GDL = self.parsed_predicate_GDL["Relation"][predicate]
        elif predicate == "Free":
            return True
        else:
            item_GDL = self.parsed_predicate_GDL["Attribution"][predicate]

        letters = {}  # used for vars-letters replacement
        for i in range(len(item_GDL["vars"])):
            letters[item_GDL["vars"][i]] = item[i]

        for name, para in item_GDL["ee_check"]:
            if tuple(letters[i] for i in para) not in self.condition.get_items_by_predicate(name):
                return False
        return True

    def fv_check(self, predicate, item):
        """Format Validity check."""

        if predicate == "Equation":
            if item is None or item == 0:
                return False
            return True
        elif predicate in self.parsed_predicate_GDL["Preset"]["Construction"]:
            if predicate == "Shape":
                if len(item) != len(set(item)):  # default check 1: mutex points
                    return False
                if len(item) == 1:
                    if len(item[0]) not in [1, 2]:
                        return False
                    return True
                for shape in item:
                    if not 2 <= len(shape) <= 3 or len(shape) != len(set(shape)):
                        return False
                return True
            else:
                return len(item) == len(set(item))  # default check 1: mutex points
        elif predicate in self.parsed_predicate_GDL["Preset"]["BasicEntity"]:
            if len(item) != len(set(item)):  # default check 1: mutex points
                return False
            if predicate == "Point" and len(item) != 1:
                return False
            elif predicate == "Line" and len(item) != 2:
                return False
            elif predicate == "Arc" and len(item) != 3:
                return False
            elif predicate == "Angle" and len(item) != 3:
                return False
            elif predicate == "Polygon" and len(item) < 3:
                return False
            elif predicate == "Circle" and len(item) != 1:
                return False
            return True
        elif predicate in self.parsed_predicate_GDL["Entity"]:
            if len(item) != len(set(item)):  # default check 1: mutex points
                return False
            item_GDL = self.parsed_predicate_GDL["Entity"][predicate]
        elif predicate in self.parsed_predicate_GDL["Relation"]:
            item_GDL = self.parsed_predicate_GDL["Relation"][predicate]
        elif predicate == "Free":
            return True
        else:
            item_GDL = self.parsed_predicate_GDL["Attribution"][predicate]

        if len(item) != len(item_GDL["vars"]):  # default check 2: correct para len
            return False

        if "fv_check" in item_GDL:  # fv check, more stringent than default check 3
            checked = []
            result = []
            for i in item:
                if i not in checked:
                    checked.append(i)
                result.append(str(checked.index(i)))
            if "".join(result) in item_GDL["fv_check"]:
                return True
            return False

        if len(item_GDL["ee_check"]) > 1:  # default check 3: para of the same type need to be different
            predicate_to_vars = {}
            for predicate, p_var in item_GDL["ee_check"]:
                if predicate not in self.parsed_predicate_GDL["Preset"]["Construction"]:  # check only BasicEntity
                    if predicate not in predicate_to_vars:
                        predicate_to_vars[predicate] = [p_var]
                    else:
                        predicate_to_vars[predicate].append(p_var)

            letters = {}  # used for vars-letters replacement
            for i in range(len(item_GDL["vars"])):
                letters[item_GDL["vars"][i]] = item[i]

            for predicate in predicate_to_vars:
                if len(predicate_to_vars[predicate]) == 1:
                    continue

                mutex_sets = []  # mutex_item
                for p_var in predicate_to_vars[predicate]:
                    mutex_sets.append([letters[i] for i in p_var])

                mutex_sets_multi = []  # mutex_item multi representation
                for mutex_item in mutex_sets:
                    if predicate == "Line":
                        mutex_sets_multi.append(tuple(mutex_item))
                        mutex_sets_multi.append(tuple(mutex_item[::-1]))
                    elif predicate == "Polygon":
                        l = len(mutex_item)
                        for bias in range(0, l):
                            mutex_sets_multi.append(tuple([mutex_item[(i + bias) % l] for i in range(l)]))
                    else:  # Point Arc Angle Circle
                        mutex_sets_multi.append(tuple(mutex_item))

                if len(mutex_sets_multi) != len(set(mutex_sets_multi)):
                    return False

        return True

    def get_sym_of_attr(self, attr, item):
        """
        Get symbolic representation of item's attribution.
        :param attr: attr's name, such as LengthOfLine
        :param item: tuple, such as ('A', 'B')
        :return: sym
        """

        if attr != "Free" and attr not in self.parsed_predicate_GDL["Attribution"]:  # attr must define
            e_msg = "Attribution '{}' not defined in current predicate GDL.".format(attr)
            raise Exception(e_msg)
        if not self.ee_check(attr, item):  # ee check
            msg = "EE check not passed: [{}, {}]".format(attr, item)
            # warnings.warn(msg)
            return None
        if not self.fv_check(attr, item):  # fv check
            msg = "FV check not passed: [{}, {}]".format(attr, item)
            # warnings.warn(msg)
            return None

        if (attr, item) in self.condition.sym_of_attr:  # already has sym
            return self.condition.sym_of_attr[(attr, item)]

        if attr == "Free":
            sym = symbols("".join(item))
            self.condition.sym_of_attr[(attr, item)] = sym  # add sym
            self.condition.value_of_sym[sym] = None  # init symbol's value
            self.condition.attr_of_sym[sym] = (attr, (item,))  # add attr
            self.condition.equivalence_of_sym[sym] = []
            return sym

        if attr == "MeasureOfAngle":  # align angle's sym
            sym = symbols("ma_" + "".join(item).lower(), positive=True)  # init sym
            self.condition.value_of_sym[sym] = None  # init symbol's value
            self.condition.sym_of_attr[("MeasureOfAngle", item)] = sym
            self.condition.attr_of_sym[sym] = ("MeasureOfAngle", (item,))
            self.condition.equivalence_of_sym[sym] = []
            # define same angle
            same_angles = self._get_same_angles(item)
            for same_angle in same_angles:
                if same_angle == item: 
                    continue
                
                # define angle
                if ("MeasureOfAngle", same_angle) not in self.condition.sym_of_attr:
                    same_angle_sym = symbols("ma_" + "".join(same_angle).lower(), positive=True)
                    self.condition.sym_of_attr[("MeasureOfAngle", same_angle)] = same_angle_sym
                    self.condition.attr_of_sym[same_angle_sym] = ("MeasureOfAngle", (same_angle,))
                    self.condition.value_of_sym[same_angle_sym] = None
                    self.condition.equivalence_of_sym[same_angle_sym] = []
                
                same_angle_sym = self.condition.sym_of_attr[("MeasureOfAngle", same_angle)]
                
                # find premise of collinear
                collinear_ps = tuple(set(item[1:] + same_angle[1:]))
                collinear_total = self.condition.get_items_by_predicate('Collinear')
                for ps in collinear_total:
                    if set(collinear_ps) == set(ps):
                        break
                collinear_premise = self.condition.get_id_by_predicate_and_item('Collinear', ps)
                
                # add equality relation
                self.condition.add('Equation', same_angle_sym - sym, [collinear_premise], ('extended', None, None))
                # self.condition.sym_of_attr[("MeasureOfAngle", same_angle)] = sym
            # self.condition.attr_of_sym[sym] = ("MeasureOfAngle", tuple(same_angles))
            
            return sym

        attr_GDL = self.parsed_predicate_GDL["Attribution"][attr]
        if (attr, item) not in self.condition.sym_of_attr:  # No symbolic representation, initialize one.
            sym = symbols(attr_GDL["sym"] + "_" + "".join(item).lower(), positive=True)
            self.condition.sym_of_attr[(attr, item)] = sym  # add sym
            self.condition.value_of_sym[sym] = None  # init symbol's value

            extend_items = [item]

            letters = {}  # used for vars-letters replacement
            for i in range(len(attr_GDL["vars"])):
                letters[attr_GDL["vars"][i]] = item[i]

            for multi in attr_GDL["multi"]:
                extended_item = [letters[i] for i in multi]  # extend item
                self.condition.sym_of_attr[(attr, tuple(extended_item))] = sym  # multi representation
                extend_items.append(tuple(extended_item))

            self.condition.attr_of_sym[sym] = (attr, tuple(extend_items))  # add attr
            self.condition.equivalence_of_sym[sym] = []
            return sym

    def set_value_of_sym(self, sym, value, premise):
        """
        Set value of sym.
        Add equation to record the premise and theorem of solving the symbol's value at the same time.
        :param sym: <symbol>
        :param value: <float>
        :param premise: tuple of <int>, premise of getting value.
        """

        if self.condition.value_of_sym[sym] is None:
            self.condition.value_of_sym[sym] = value
            if 'ma_eac' in str(sym):
                a = 1 # debug
            added, _id = self.condition.add("Equation", sym - value, premise, ("solve_eq", 'set_value', None))
            return added
        return False

    def step(self, item, timing):
        """
        Execute when theorem successful applied. Save theorem and update step.
        :param item: <str>, theorem, 'init_problem' and 'check_goal'.
        :param timing: <float>.
        """
        self.timing[self.condition.step_count] = (item, timing)
        self.condition.step()

    def check_goal(self):
        """Check whether the solution is completed."""
        s_start_time = time.time()  # timing
        if self.goal.type == "algebra":  # algebra relation
            try:
                result, premise = EqKiller.solve_target(self.goal.item, self)
            except BaseException as e:
                msg = "Exception occur when solve target: {}".format(repr(e))
                warnings.warn(msg)
            else:
                if result is not None:
                    if rough_equal(result, self.goal.answer):
                        self.goal.solved = True
                    self.goal.solved_answer = result

                    eq = self.goal.item - result
                    if eq in self.condition.get_items_by_predicate("Equation"):
                        self.goal.premise = self.condition.get_premise_by_predicate_and_item("Equation", eq)
                        self.goal.theorem = self.condition.get_theorem_by_predicate_and_item("Equation", eq)
                    else:
                        self.goal.premise = tuple(premise)
                        self.goal.theorem = ("solve_eq", None, None)
        elif self.goal.type == "logic":  # logic relation
            if self.goal.answer in self.condition.get_items_by_predicate(self.goal.item):
                self.goal.solved = True
                self.goal.solved_answer = self.goal.answer
                self.goal.premise = self.condition.get_premise_by_predicate_and_item(self.goal.item, self.goal.answer)
                self.goal.theorem = self.condition.get_theorem_by_predicate_and_item(self.goal.item, self.goal.answer)

        # self.step("check_goal", time.time() - s_start_time)
