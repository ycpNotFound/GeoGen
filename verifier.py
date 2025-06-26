import itertools
import json
import os
import numpy as np
import math
from copy import deepcopy

from sympy import Eq, cos, rad, simplify, sin, solve, tan

# from formalgeo.solver import Interactor
from formalgeo_v2.parse.basic import (
    get_equation_from_tree,
    get_expr_from_tree,
    parse_equal_predicate,
    parse_geo_predicate,
)
from solver import Interactor
from utils.formulate import clause_to_nature_language
from utils.symbolic import build_point_map, parse_clause, replace_points_for_clause
from utils.tools import read_json
from tqdm import tqdm

ENTITIES = [
    "Triangle",
    "RightTriangle",
    "EquilateralTriangle",
    "IsoscelesTriangle",
    "IsoscelesRightTriangle",
    "Parallelogram",
    "Rectangle",
    "Rhombus",
    "Square",
    "Trapezoid",
    "Kite",
    "RightTrapezoid",
    "IsoscelesTrapezoid",
]

PREDICATES_REL_2 = [
    'SimilarBetweenTriangle', 'SimilarBetweenQuadrilateral',
    'CongruentBetweenTriangle', 'CongruentBetweenQuadrilateral',
    'MirrorSimilarBetweenTriangle', 'MirrorCongruentBetweenTriangle',
    'MirrorCongruentBetweenQuadrilateral'
]

PARA_LEN = {
    'll_': 2, 'ma_': 3, 'pt_': 3, 'at_': 3, 'ma_3': 3, 'rst_': 6, 'mnt_': 6,
    'pq_': 4, 'aq_': 4, 'hq_': 4, 'rsq_': 8, 'rmq_': 8, 'la_': 3, 'mar_': 3,
    'rsa_': 3, 'rc_': 1, 'dc_': 1, 'pc_': 1, 'ac_': 1, 'ps_': 3, 'as_': 3,
}

def rotate_combinations(lst):
    n = len(lst)
    for i in range(n):
        rotated = lst[i:] + lst[:i]
        yield rotated


class Verifier():
    def __init__(self, predicate_GDL, theorem_GDL, t_info, natural_template, strict=False, debug=False):
        self.predicate_GDL = predicate_GDL
        self.theorem_GDL = theorem_GDL
        self.t_info = t_info
        self.natural_template = natural_template
        self.strict = strict
        self.debug = debug

        # build map: predicate to extend conclusions
        self.predicate_to_extend_map = {}
        for template, info in self.predicate_GDL["Entity"].items():
            name = template.split("(")[0]
            self.predicate_to_extend_map[name] = {
                'template': template,
                'extend': info['extend'],
                'multi': info['multi']
            }

        for template, info in self.predicate_GDL["Relation"].items():
            name = template.split("(")[0]
            self.predicate_to_extend_map[name] = {
                'template': template,
                'extend': info['extend'],
                'multi': info['multi']
            }

        self.attr_dict = self.build_attr_dict()

    def sort_counter_clockwise_arc(self, c, pa, pb):
        # 向量OA和OB
        if self.p_pos is None or any(p not in self.p_pos for p in [c, pa, pb]):
            return (pa, pb)
        
        xa, ya = self.p_pos[pa]
        xb, yb = self.p_pos[pb]
        xo, yo = self.p_pos[c]
        radius_2 = (xo - xa) ** 2 + (yo - ya) ** 2

        OA = np.array([xa, ya]) - np.array([xo, yo])
        OB = np.array([xb, yb]) - np.array([xo, yo])

        # 计算两个向量的夹角（点积公式）
        dot_product = np.dot(OA, OB)
        angle_cos = dot_product / radius_2
        angle = np.arccos(np.clip(angle_cos, -1.0, 1.0))  # 使用clip防止数值误差导致acos超出范围

        # 判断角度的方向（通过叉积判断）
        cross_product_z = np.cross(OA, OB)

        if cross_product_z < 0:
            # 逆时针方向较小的角（劣弧）a -> b
            return (pa, pb)
        else:
            # 逆时针方向较大的角（优弧），应该反向
            return (pb, pa)

    def sort_counter_clockwise_angle(self, points):
        if self.p_pos is None or any(p not in self.p_pos for p in points):
            return points

        (A, B, C) = [self.p_pos[p] for p in points]
        AB = (B[0] - A[0], B[1] - A[1])
        BC = (C[0] - B[0], C[1] - B[1])
        cross_product_z = AB[0]*BC[1] - AB[1]*BC[0]
        if cross_product_z >= 0:
            return tuple([points[2], points[1], points[0]])
        else:
            return points

    def sort_counter_clockwise_cocircular(self, centre, points):
        if self.p_pos is None or centre not in self.p_pos or any(p not in self.p_pos for p in points):
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

    def sort_counter_clockwise(self, points):
        if self.p_pos is None or any(p not in self.p_pos for p in points):
            return points
        xo = sum([self.p_pos[p][0] for p in points]) / len(points)
        yo = sum([self.p_pos[p][1] for p in points]) / len(points)

        def angle_from_origin(point):
            dx = point[0] - xo
            dy = point[1] - yo
            return math.atan2(dy, dx)

        angles = [(key, angle_from_origin(self.p_pos[key])) for key in points]
        sorted_angles = sorted(angles, key=lambda x: x[1], reverse=True)
        sorted_points = [item[0] for item in sorted_angles]
        start_idx = sorted_points.index(min(points))
        sorted_points = sorted_points[start_idx:] + sorted_points[:start_idx]
        return tuple(sorted_points)

    def sort_by_x_collinear(self, points):
        if self.p_pos is None or any(p not in self.p_pos for p in points):
            return points
        x1, x2 = self.p_pos[points[0]][0], self.p_pos[points[1]][0]
        if abs(x1 - x2) < 1:
            sorted_points = sorted(points, key=lambda x: self.p_pos[x][1])
        else:
            sorted_points = sorted(points, key=lambda x: self.p_pos[x][0])
        return tuple(sorted_points)

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

    def build_attr_dict(self):
        attr_dict = {}
        for k, v in self.predicate_GDL['Attribution'].items():
            sym = v['sym'] + '_'
            predicate = k.split('(')[0]
            attr_dict[sym] = predicate

        return attr_dict

    def init_problem(self, text_cdls, constr_cdls, image_cdls, p_pos=None, lines=None):
        self.p_pos = p_pos
        self.solver = Interactor(
            self.predicate_GDL,
            self.theorem_GDL,
            p_pos=p_pos,
            t_info=self.t_info,
            debug=self.debug
        )
        self.problem_CDL = {
            "problem_id": 0,
            "construction_cdl": constr_cdls,
            "text_cdl": text_cdls,
            "image_cdl": image_cdls,
            "goal_cdl": ["Value(0)"],
            "problem_answer": 0,
            "point_positions": p_pos if p_pos else {},
            "line_instances": lines if lines else [],
        }

        self.solver.load_problem(self.problem_CDL, solve_eq=self.strict)
        self.predicate_names = list(self.solver.problem.condition.items_group.keys())
        self.predicate_names.remove('Equation')

    def check_equations(self, eq_expr, equations):
        for eq_i in equations:
            res_expr = simplify(f"(({str(eq_expr)}) - ({str(eq_i)}))")
            if res_expr == 0:
                return True
        return False

    def check_paras(self, self_, name, para, items):
        if type(para) != tuple:
            para = tuple(para)
        if name in ['Line', 'Shape', 'Polygon', 'Collinear', 'Triangle']:
            # No location restriction
            para_permutations = list(itertools.permutations(para))
            if any([para_1 in items for para_1 in para_permutations]):
                return True
        elif name in ['EquilateralTriangle', 'Parallelogram', 'Kite', 'Rhombus', 'Rectangle', 
                    'Square', 'Trapezoid', 'IsoscelesTrapezoid']:
            # Can be rotation
            para_rotations = rotate_combinations(para)
            if any([para_1 in items for para_1 in para_rotations]):
                return True
        else:
            # Other cases
            if para in items:
                return True

        return False

    def get_equation_from_tree(self, tree):
        left_expr = get_expr_from_tree(
            self.solver.problem, tree[0], False, None
        )
        if left_expr is None:
            # try expand symbol
            if type(tree[0]) == tuple:
                if tree[0][1] == 'MeasureOfAngle':
                    left_expr = tree[0][1][-1::-1]
                elif tree[0][1] == 'MeasureOfArc':
                    if len(self.solver.problem.condition.items_group['Circle']) == 0:
                        return None
                    circle = self.solver.problem.condition.items_group['Circle'][0][0]
                    left_points = tuple([circle, tree[0][1][2], tree[0][1][1]])
                    return None
                else:
                    return None
        left_expr = get_expr_from_tree(
            self.solver.problem, (tree[0][0], left_points), False, None
        )
        if left_expr is None:
            return None

        right_expr = get_expr_from_tree(
            self.solver.problem, tree[1], False, None
        )
        if right_expr is None:
            # try expand symbol
            if type(tree[1]) == tuple:
                if tree[1][0] == 'MeasureOfAngle':
                    right_points = tree[1][1][::-1]
                elif tree[1][0] == 'MeasureOfArc':
                    if len(self.solver.problem.condition.items_group['Circle']) == 0:
                        return None
                    circle = self.solver.problem.condition.items_group['Circle'][0][0]
                    right_points = tuple([circle, tree[1][1][2], tree[1][1][1]])
                else:
                    return None
                right_expr = get_expr_from_tree(
                    self.solver.problem, (tree[1][0], right_points), False, None
                )
                if right_expr is None:
                    return None

        return left_expr - right_expr

    def clause_to_expr_str(self, clause):
        name = clause.split('(')[0]
        assert name in ['Value', 'Equal', 'Equation']
        if name in ['Value', 'Equal']:
            clause = clause.replace('Value', 'Equal')
            try:
                eq_tree, attrs = parse_equal_predicate(clause)
                eq_expr = self.get_equation_from_tree(eq_tree[1])
            except:
                try:
                    eq_tree, attrs = parse_equal_predicate(clause + ')')
                    eq_expr = self.get_equation_from_tree(eq_tree[1])
                except:
                    return None
        elif name == 'Equation':
            eq_expr = clause.replace('Equation(', '')
            eq_expr = eq_expr[:-1]

        return eq_expr if eq_expr else None

    def verify_existance(self, conditions):
        statements = clause_to_nature_language(
            conditions,
            natural_template=self.natural_template
        )
        for clause, statement in zip(conditions, statements):
            name = clause.split('(')[0]
            if name in self.predicate_names:
                items = self.solver.problem.condition.get_items_by_predicate(name)
                _, para, _ = parse_geo_predicate(clause)
                if not self.check_paras(name, para, items):
                    return f"Relation: {statement} doesn't exist."
            if not self.check_paras(name, para, items):
                msg = f"Relation: {statement} doesn't exist."
                return False, msg

            elif name in ['Value', 'Equal', 'Equation']:
                eq_str = self.clause_to_expr_str(clause)
                equations = [item[1] for item in self.solver.problem.condition.items if item[0] == 'Equation']
                if not self.check_equations(eq_str, equations):
                    msg = f"Relation: {statement} doesn't exist."
                    return False, msg
            else:
                raise KeyError(name)

        return True, None

    def extend_conditions(self, conditions):
        conclusions_total = []
        for clause in conditions:
            name = clause.split('(')[0]
            if name in self.predicate_to_extend_map:
                # extend origin clause
                conclusions_total += self.extend_one_condition(clause)

                # extend multiple clause
                clause_template = self.predicate_to_extend_map[name]['template']
                p_map = build_point_map(clause, clause_template)
                # multi: ABCD -> BCDA, CDAB, DABC
                for para_str in self.predicate_to_extend_map[name]['multi']:
                    new_clause = f"{name}({para_str})"
                    new_clause = replace_points_for_clause(new_clause, p_map)
                    conclusions_total += self.extend_one_condition(new_clause)

            elif name in ['Equal', 'Value', 'Equation']:
                continue
            else:
                raise KeyError(name)
        return sorted(set(conclusions_total))

    def extend_one_condition(self, clause):
        extend_total = []
        name = clause.split('(')[0]
        if name in self.predicate_to_extend_map:
            clause_template = self.predicate_to_extend_map[name]['template']
            extends_template = self.predicate_to_extend_map[name]['extend']
            # template -> clause, use recursion
            p_map = build_point_map(clause, clause_template)
            for extend_temp in extends_template:
                extend_clause = replace_points_for_clause(extend_temp, p_map)
                extend_total.append(extend_clause)
                extend_total += self.extend_one_condition(extend_clause)

        elif name in ['Equal', 'Value', 'Equation']:
            return []
        else:
            raise KeyError(name)

        return extend_total
    
    def verify_theorem(self, theorem, conditions, conclusions):
        cdtn_statements = clause_to_nature_language(
            conditions,
            natural_template=self.natural_template
        )
        cclsn_statements = clause_to_nature_language(
            conclusions,
            natural_template=self.natural_template
        )

        if theorem is None:
            # extend all conclusions first
            conclusions_total = self.extend_conditions(conditions)

            # then check conclusion existence
            for cclsn, statement in zip(conclusions, cclsn_statements):
                cclsn = cclsn.replace('Value', 'Equal')
                if cclsn not in conclusions_total:
                    msg = f"Conclusion {statement} can't be inferred through conditions:\n{' ,'.join(cdtn_statements)}."
                    return False, msg

        elif theorem == 'solve_eq':
            eq_expr_lst, cclsn_expr_lst = [], []
            for clause in conditions:
                name = clause.split('(')[0]
                if name in ['Value', 'Equal', 'Equation']:
                    eq_str = self.clause_to_expr_str(clause)
                    eq_expr_lst.append(eq_str)
                else:
                    raise KeyError(name)

            for clause in conclusions:
                name = clause.split('(')[0]
                if name in ['Value', 'Equal', 'Equation']:
                    eq_str = self.clause_to_expr_str(clause)
                    cclsn_expr_lst.append(eq_str)
                else:
                    raise KeyError(name)

            eqs = [Eq(simplify(expr), 0) for expr in eq_expr_lst]
            solution = solve(eqs, dict=True)

            if len(solution) == 0:
                # equation group can't be solved
                eq_str = '\n'.join(cdtn_statements)
                verify_msg = f"Equations group:\n{eq_str}\ncan't be solved."
                return False, verify_msg

            else:
                if isinstance(solution, list):
                    solution = solution[0]
                results_eqs = [sym - value for sym, value in solution.items()]
                # check conclusions in solved results
                for expr, statement in zip(cclsn_expr_lst, cclsn_statements):
                    if not self.check_equations(expr, results_eqs):
                        condition_eq_str = '\n'.join(cdtn_statements)
                        verify_msg = f"Conclusion {statement} can't be solved from equations group:\n{condition_eq_str}"
                        return False, verify_msg
                    
        elif theorem not in self.solver.parsed_theorem_GDL:
            added_theorems = ['cos_of_angle', 'sin_of_angle', 'tan_of_angle']

            self.solver.solve_special_angles()
            if theorem in added_theorems:
                angle_chars = [parse_clause(c)[1][0] for c in conditions]
                angle_values = [int(parse_clause(c)[1][1]) for c in conditions]

                special_angle = None
                if 60 in angle_values and 90 in angle_values:
                    special_angle = 60
                elif 30 in angle_values and 90 in angle_values:
                    special_angle = 30
                elif 45 in angle_values and 90 in angle_values:
                    special_angle = 45

                if special_angle is not None:
                    angle_sp = angle_chars[angle_values.index(special_angle)]
                    angle_90 = angle_chars[angle_values.index(90)]

                    # l1 = cos(a) * l1
                    # l2 = sin(a) * l1
                    # l3 = tan(a) * l1
                    l1 = (angle_sp[0], angle_sp[1], angle_sp[1])
                    l2 = (angle_sp[2], angle_sp[1], angle_sp[0])
                    l3 = (angle_sp[2], angle_sp[0], angle_sp[1])
                    l1, l2, l3 = angle_90[1], angle_90[2], angle_90[0]

                    l1_sym = self.solver.problem.get_sym_of_attr('LengthOfLine', l1)
                    l2_sym = self.solver.problem.get_sym_of_attr('LengthOfLine', l2)
                    l3_sym = self.solver.problem.get_sym_of_attr('LengthOfLine', l3)

                    expr_1 = l2_sym - cos(rad(special_angle)) * l1_sym
                    expr_2 = l3_sym - sin(rad(special_angle)) * l1_sym
                    expr_3 = l3_sym - tan(rad(special_angle)) * l2_sym
                    expr_lst = [expr_1, expr_2, expr_3]
                    expr = expr_lst[added_theorems.index(theorem)]

                    eq_str = self.clause_to_expr_str(conclusions[0])
                    if not self.check_equations(eq_str, [expr]):
                        verify_msg = f"Conclusion {cclsn_statements[0]} can't be solved from {{theorem}}: {' ,'.join(cdtn_statements)}"
                        return False, verify_msg
            else:
                raise KeyError(theorem)
        else:
            # apply solver directly
            update = self.solver.apply_theorem_by_name(theorem)
            if not update:
                verify_msg = f"Theorem {{theorem}} can't be applied."
                return False, verify_msg

        return True, None

    def verify_strict(self, solution_dict):
        for step, info in solution_dict.items():
            theorem = info['theorem']
            conditions = info['condition']
            conclusions = info['conclusion']

            # check condition existence
            verify_flag, verify_msg = self.verify_existance(conditions)
            if not verify_flag:
                return False, verify_msg

            # check application of theorem
            verify_flag, verify_msg = self.verify_theorem(theorem, conditions, conclusions)
            if not verify_flag:
                return False, verify_msg

            # check conclusion existence
            verify_flag, verify_msg = self.verify_existance(conclusions)
            if not verify_flag:
                return False, verify_msg

        return True, None

    def sort_points_for_equivalce(self, name, para):
        if name == 'Collinear':
            para = self.sort_by_x_collinear(para)
        elif name == 'Cocircular':
            para = self.sort_counter_clockwise_cocircular(para)
        elif name == 'ParallelBetweenLine':
            para, _ = self.same_side_for_parallel(para)
        elif name == 'IsBisectorOfAngle':
            para = para[:2] + list(self.sort_counter_clockwise_angle(para[2:]))
        elif 'Congruent' in name or 'Similar' in name:
            item_1, item_2 = para[:len(para)//2], para[len(para)//2:]
            item_1_ccw = list(self.sort_counter_clockwise(item_1))
            item_2_ccw = list(self.sort_counter_clockwise(item_2))
            if item_1 == item_1_ccw and item_2 == item_2_ccw:
                name = f"Mirror{name}"
            elif item_1 == item_2_ccw and item_2 != item_2_ccw:
                name = f"Mirror{name}"
            para = item_1_ccw + item_2_ccw

        return name, para

    def expand_for_equivalence(self, name, para):
        name_paras = [(name, para)]

        if name in PREDICATES_REL_2:
            item_1, item_2 = para[:len(para)//2], para[len(para)//2:]
            item_1_inv = item_1[:1] + item_1[1:][::-1]
            item_2_inv = item_2[:1] + item_2[1:][::-1]

            if 'Mirror' in name:
                new_name = name.replace('Mirror', '')
                name_paras += [
                    (new_name, item_1 + item_2),
                    (new_name, item_1_inv + item_2),
                    (new_name, item_1_inv + item_2_inv),
                    (new_name, item_1 + item_2_inv),
                    (name, item_1 + item_2),
                    (name, item_1_inv + item_2_inv),
                ]
            else:
                name_paras += [
                    (name, item_1 + item_2_inv),
                    (name, item_1_inv + item_2),
                    (name, item_1_inv + item_2_inv),
                    (f"Mirror{name}", item_1 + item_2_inv),
                    (f"Mirror{name}", item_1_inv + item_2),
                    (f"Mirror{name}", item_1 + item_2),
                    (f"Mirror{name}", item_1_inv + item_2_inv),
                ]

        elif name in ENTITIES:
            # ABCD -> ADCB
            para_inv = para[1:] + para[1:][::-1]
            name_paras.append((name, para_inv))

        elif name == 'IsMidpointOfArc':
            p, arc = para[:1], para[1:]
            para_inv = p + arc[1:] + arc[1:][::-1]
            name_paras.append((name, para_inv))

        elif name in ['ParallelBetweenLine', 'PerpendicularBetweenLine']:
            # AB // CD -> AB // CD, BA // CD, BA // DC, CD // AB, CD // BA, DC // AB, DC // BA
            l1, l2 = para[:2], para[2:]
            paras = [
                l1 + l2[:1], l1[::-1] + l2, l1[::-1] + l2[::-1],
                l2 + l1, l2[::-1] + l1, l2[::-1] + l1[::-1],
            ]
            name_paras += [(name, para) for para in paras]

        elif name == 'IsBisectorOfAngle':
            line, angle = para[:2], para[2:]
            paras = [
                line[::-1] + angle,
                line + angle[::-1],
                line[::-1] + angle[::-1]
            ]
            name_paras += [(name, para) for para in paras]


        elif name == 'IsPerpendicularBisectorOfLine':
            l1, l2 = para[:2], para[2:]
            paras = [
                l1 + l2[::-1], l1[::-1] + l2, l1[::-1] + l2[::-1]
            ]
            name_paras += [(name, para) for para in paras]

        elif name in ['IsMedianOfTriangle', 'IsAltitudeOfTriangle', 'IsMidsegmentOfTriangle', 'IsMidsegmentOfQuadrilateral']:
            tri, line = para[:2], para[2:]
            paras = [
                line[::-1] + tri, line + tri[::-1], line[::-1] + tri[::-1]
            ]
            name_paras += [(name, para) for para in paras]

        elif name in ['IsCircumcenterOfTriangle', 'IsIncenterOfTriangle', 'IsCentroidOfTriangle',
                    'IsOrthocenterOfTriangle', 'IsCircumcenterOfQuadrilateral', 'IsIncenterOfQuadrilateral']:
            p, tri = para[:1], para[1:]
            paras = [
                p + tri[::-1]
            ]
            name_paras += [(name, para) for para in paras]

        elif name == 'CongruentBetweenArc':
            if len(self.solver.problem.condition.items_group['Circle']) == 0:
                return name_paras
            circle = self.solver.problem.condition.items_group['Circle'][0][0]
            item_1, item_2 = (circle,) + para[1:3], (circle,) + para[4:6]
            item_1_inv = item_1[:1] + item_1[1:][::-1]
            item_2_inv = item_2[:1] + item_2[1:][::-1]
            name_paras += [
                (name, item_1 + item_2),
                (name, item_1_inv + item_2),
                (name, item_1 + item_2_inv),
                (name, item_1_inv + item_2_inv),
            ]

        return name_paras

    def expand_for_symbol(self, sym):
        syms_expand = []
        if 'ma_' in sym:
            points = [c for c in sym.split("ma_")[1]]
            sym = "ma_" + ''.join(points[::-1]).lower()
            syms_expand.append(sym)
        elif 'fma_' in sym:
            points = [c for c in sym.split("fma_")[1]]
            if len(self.solver.problem.condition.items_group['Circle']) == 0:
                return syms_expand
            circle = self.solver.problem.condition.items_group['Circle'][0][0].lower()
            new_points = [circle] + points[1:][::-1]
            sym = f"fma_{''.join(new_points).lower()}"
            syms_expand.append(sym)
        elif 'll_' in sym:
            points = [c for c in sym.split("ll_")[1]]
            sym = "ll_" + ''.join(points[::-1]).lower()
            syms_expand.append(sym)
        elif sym in ['x', 'c', 'k', 'y', 'z']:
            return [sym]
        else:
            return []

        return syms_expand
    
    def verify_one_para(self, name, para):
        if not self.solver.problem.fv_check(name, para):  # fv check
            w_msg = f"FV check not passed: [{name}, {para}]"
            return False, w_msg

        if not self.solver.problem.eq_check(name, para):  # eq check
            w_msg = f"EQ check not passed: [{name}, {para}]"
            return False, w_msg

        return True, None

    def verify_premise_of_conditions(self, conditions):
        for clause in conditions:
            name = clause.split('(')[0]
            if name in self.predicate_names:
                _, para, _ = parse_geo_predicate(clause)
                # handle equivalence in geometry
                # name, para = self.sort_points_for_equivalce(name, para)

                para = self.sort_points_for_equivalce(name, para)[1]
                items = self.solver.problem.condition.get_items_by_predicate(name)

                para = tuple(para)
                # if has existed in conditions
                if self.check_paras(name, para, items):
                    continue

                name_paras_expand = self.expand_for_equivalence(name, para)
                # first check
                res_i_list = []
                msg_i_list = []
                for name, para in name_paras_expand:
                    res, msg = self.verify_one_para(name, para)
                    res_i_list.append(res)
                    msg_i_list.append(msg)
                if not any(res_i_list):
                    return False, msg_i_list[0]

                correct_idx = res_i_list.index(True)
                name_target, para_target = name_paras_expand[correct_idx]
                # self.solver.problem.add(
                #     name_target, para_target, (-1, ), ('prerequisite', None, None), skip_check=True
                # )

            elif name in ['Value', 'Equal', 'Equation']:
                eq_str = self.clause_to_expr_str(clause)
                if not eq_str:
                    msg = f"Attribution of clause {clause} doesn't exist."
                    return False, msg
                try:
                    symbols = [str(s) for s in list(simplify(eq_str).free_symbols)]
                except:
                    msg = f"Can not parse expressions from clause {clause}."
                    return False, msg
                for sym in symbols:
                    # convert symbol to ccw for angle / arc
                    sym_expand = self.expand_for_symbol(sym)
                    res_i_list = []
                    for sym_i in sym_expand:
                        res_i = self.verify_symbol_existance(sym_i)
                        res_i_list.append(res_i)

                    if not any(res_i_list):
                        msg = f"{sym} is not in existed symbols."
                        return False, msg

            else:
                msg = f"{name} is not defined."
                return False, msg
            
        return True, None

    def verify_symbol_existance(self, sym_str):
        if str(sym_str) in [str(s) for s in self.solver.problem.condition.value_of_sym]:
            return True

        for sym_prefix, predicate in self.attr_dict.items():
            if sym_prefix in sym_str:
                para = tuple(sym_str.split(sym_prefix)[-1].upper())
                if len(para) != PARA_LEN[sym_prefix]:
                    return False

                res = self.solver.problem.ee_check(predicate, para)
                if not res:
                    return False
                else:
                    self.solver.problem.get_sym_of_attr(predicate, para)
        return True

    def check_para_len(self, name, para):
        if name in self.solver.parsed_predicate_GDL['Entity']:
            predicate_info = self.solver.parsed_predicate_GDL['Entity'][name]
        elif name in self.solver.parsed_predicate_GDL['Relation']:
            predicate_info = self.solver.parsed_predicate_GDL['Relation'][name]
        elif name in self.solver.parsed_predicate_GDL['Attribution']:
            predicate_info = self.solver.parsed_predicate_GDL['Attribution'][name]
        elif name == 'Line':
            return len(para) == 2
        elif name == 'Angle':
            return len(para) == 3
        else:
            return True

        res = len(para) == sum(predicate_info['para_len'])
        return res

    def add_clause_to_condition(self, clause):
        name = clause.split('(')[0]
        if name in self.predicate_names:
            _, para, _ = parse_geo_predicate(clause)
            items = self.solver.problem.condition.get_items_by_predicate(name)

            para = tuple(para)
            # if has existed in conditions
            if self.check_paras(name, para, items):
                return
            else:
                if self.check_para_len(name, para):
                    self.solver.problem.add(
                        name, para, (-1, ), ('prerequisite', None, None), skip_check=True
                    )
        elif name in ['Value', 'Equal']:
            clause = clause.replace('Value', 'Equal')
            attr_name, paras = parse_clause(clause)
            if attr_name == 'LengthOfLine':
                pred_to_add = 'Line'
            elif attr_name == 'MeasureOfAngle':
                pred_to_add = 'Angle'
            else:
                return
            for para in paras:
                if not para.isupper() or para.isdigit():
                    continue
                if self.check_para_len(attr_name, tuple(para)):
                    self.solver.problem.add(
                        pred_to_add, tuple(para), (-1, ), ('prerequisite', None, None), skip_check=True
                    )
        return

    def add_condition_in_assume(self, assume_step):
        for clause in assume_step['condition']:
            self.add_clause_to_condition(clause)
        for clause in assume_step['conclusion']:
            self.add_clause_to_condition(clause)

    def verify_premise(self, solution_dict, solution_str_dict):
        steps_res = []
        for step, info in solution_dict.items():
            if step == '0':
                if 'assume' in solution_str_dict[step]:
                    self.add_condition_in_assume(info)
                    steps_res.append(True)
                    continue

            res_step_i = True
            conditions = info['condition']
            conclusions = info['conclusion']

            # check condition existence
            verify_flag, verify_msg = self.verify_premise_of_conditions(conditions)
            if not verify_flag:
                res_step_i = False
                self.verify_premise_of_conditions(conditions)

            # check conclusion existence
            verify_flag, verify_msg = self.verify_premise_of_conditions(conclusions)
            if not verify_flag:
                res_step_i = False
                self.verify_premise_of_conditions(conclusions)

            steps_res.append(res_step_i)

        return steps_res

    def verify(self, solution_dict, solution_str_dict=None):
        if self.strict:
            return self.verify_strict(solution_dict)
        else:
            return self.verify_premise(solution_dict, solution_str_dict)
