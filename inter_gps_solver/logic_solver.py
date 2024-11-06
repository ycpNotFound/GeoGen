import math
import random

from .utils_solver import isNumber, hasNumber, heron_triangle_formula, angle_area_formula
from .logic_parser import LogicParser
from .extended_definition import ExtendedDefinition

# from sympy import Symbol, Number
# from sympy import cos, sin, pi, solve, nonlinsolve

from itertools import permutations, product, combinations

from func_timeout import func_timeout, FunctionTimedOut

class LogicSolver:
    def __init__(self, logic):
        self.logic = logic
        self.can_search = False
        self.hasSolution = False
        self.equations = []
        self.function_maps = {1: self.func1_direct_triangle_sum_theorem, 2: self.func2_indirect_triangle_sum_theorem,
                       3: self.func3_isosceles_triangle_theorem_line, 4: self.func4_isosceles_triangle_theorem_angle,
                       5: self.func5_congruent_triangles_theorem_line, 6: self.func6_congruent_triangles_theorem_angle,
                       7: self.func7_radius_equal_theorem, 8: self.func8_tangent_radius_theorem,
                       9: self.func9_center_and_circumference_angle, 10: self.func10_parallel_lines_theorem,
                       11: self.func11_flat_angle_theorem, 12: self.func12_intersecting_chord_theorem,
                       13: self.func13_polygon_interior_angles_theorem, 14: self.func14_similar_triangle_theorem,
                       15: self.func15_angle_bisector_theorem};
        self.interval = 11
        # <= interval: Propagate;   > interval: Build Equations.

    @staticmethod
    def _triangleEqual(length, angle, original_angle):
        """
        Please consider the order in the parameters
        length[0..5](Boolean)   length[ch]<->length[ch+3]
        angle[0..5] (Boolean)   angle[ch] <-> angle[ch+3]
        The order of original_angle[0..5](list) matters.
        """
        if sum(length) >= 3 or sum(length) >= 1 and sum(angle) >= 2:
            return True  # SSS or AAS
        if sum(length) >= 2 and sum(angle) == 1:
            if all([angle[0], length[0], length[1]]) or all([angle[1], length[0], length[2]]) or all(
                    [angle[2], length[0], length[1]]):
                return True  # SAS
            id = angle.index(True)
            if 90 in original_angle[id] or 90 in original_angle[id + 3]:
                return True  # HL
        return False

    @staticmethod
    def _traingleSimilar(angle):
        return sum(angle) >= 2

    @staticmethod
    def _hasSymbol(expr):
        if type(expr) in [int, float]:
            return False
        return len(expr.free_symbols) > 0

    @staticmethod
    def _generateAngles(tri):
        return [(tri[0], tri[2], tri[1]), (tri[0], tri[1], tri[2]), (tri[1], tri[0], tri[2])]

    @staticmethod
    def _generateLines(tri):
        return [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]

    @staticmethod
    def _isComplex(st):
        return st.find("sin") != -1 or st.find("cos") != -1 or st.find("**2") != -1

    @staticmethod
    def _isTrig(st):
        return st.find("sin") != -1 or st.find("cos") != -1

    @staticmethod
    def _same(list1, list2):
        return any([pair[0] == pair[1] for pair in product(list1, list2)])

    def _isPrimitive(self, expr):
        return self._hasSymbol(expr) and all([str(expr).find(t) == -1 for t in ['+', '-', '*']])

   

    def func1_direct_triangle_sum_theorem(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            measures = [self.logic.find_angle_measure(x) for x in angles]
            unknowns = [i for i in range(3) if not hasNumber(measures[i])]
            if 1 <= len(unknowns) <= 2:
                idx = unknowns[0]
                other1, other2 = measures[(idx + 1) % 3][0], measures[(idx + 2) % 3][0]
                Update = self.logic.define_angle_measure(*angles[idx], 180 - other1 - other2) or Update
        return Update

    def func2_indirect_triangle_sum_theorem(self):
        # The sum of three angles in a triangle is 180
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            measures = [self.logic.find_angle_measure(x) for x in angles]
            unknowns = [i for i in range(3) if not hasNumber(measures[i])]
            if len(unknowns) == 3:
                Update = self.logic.define_angle_measure(*angles[0], 180 - measures[1][0] - measures[2][0]) or Update
        return Update

    def func3_isosceles_triangle_theorem_line(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            measures = [self.logic.find_angle_measure(x) for x in angles]
            lines = self._generateLines(tri)
            for ch in permutations([0, 1, 2]):
                if self._same(measures[ch[0]], measures[ch[1]]):
                    Update = self.logic.lineEqual(lines[ch[0]], lines[ch[1]]) or Update
        return Update

    def func4_isosceles_triangle_theorem_angle(self):
        # The base angles of a isosceles triangle is equal.
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            lines = self._generateLines(tri)
            length = [self.logic.find_line_with_length(x) for x in lines]
            for ch in permutations([0, 1, 2]):
                if self._same(length[ch[0]], length[ch[1]]):
                    Update = self.logic.angleEqual(angles[ch[0]], angles[ch[1]]) or Update
        return Update

    def func5_congruent_triangles_theorem_line(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        comb = combinations(triangles, 2)
        for pair in comb:
            for tri1 in permutations(pair[0]):
                tri2 = pair[1]
                lines = self._generateLines(tri1) + self._generateLines(tri2)
                length = [self.logic.find_line_with_length(x) for x in lines]
                angles = [self.logic.find_angle_measure(x) for x in
                          self._generateAngles(tri1) + self._generateAngles(tri2)]
                s = self._same
                same_length = [s(length[0], length[3]), s(length[1], length[4]), s(length[2], length[5])]
                same_angle = [s(angles[0], angles[3]), s(angles[1], angles[4]), s(angles[2], angles[5])]
                if self._triangleEqual(same_length, same_angle, angles):
                    for ch in range(3):
                        Update = self.logic.lineEqual(lines[ch], lines[ch + 3]) or Update
        return Update

    def func6_congruent_triangles_theorem_angle(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        comb = combinations(triangles, 2)
        for pair in comb:
            for tri1 in permutations(pair[0]):
                tri2 = pair[1]
                lines = self._generateLines(tri1) + self._generateLines(tri2)
                length = [self.logic.find_line_with_length(x) for x in lines]
                angles = [self.logic.find_angle_measure(x) for x in
                          self._generateAngles(tri1) + self._generateAngles(tri2)]
                s = self._same
                same_length = [s(length[0], length[3]), s(length[1], length[4]), s(length[2], length[5])]
                same_angle = [s(angles[0], angles[3]), s(angles[1], angles[4]), s(angles[2], angles[5])]
                if self._triangleEqual(same_length, same_angle, angles):
                    for ch in range(3):
                        Update = self.logic.angleEqual(self._generateAngles(tri1)[ch],
                                                       self._generateAngles(tri2)[ch]) or Update
        return Update

    def func7_radius_equal_theorem(self):
        Update = False
        circles = self.logic.find_all_circles()
        for circle in circles:
            points = self.logic.find_points_on_circle(circle)
            if len(points) > 1:
                # The length of each radius is same.
                for i in range(len(points) - 1):
                    Update = self.logic.lineEqual((circle, points[i]), (circle, points[i + 1])) or Update
        return Update

    def func8_tangent_radius_theorem(self):
        Update = False
        circles = self.logic.find_all_circles()
        for circle in circles:
            points = self.logic.find_points_on_circle(circle)
            for p1, p2, p3 in permutations(points, 3):
                if self.logic.find_angle_measure((p1, circle, p3))[0] == 180 and self.logic.check_angle((p1, p2, p3)):
                    Update = self.logic.define_angle_measure(p1, p2, p3, 90) or Update
        return Update

    def func9_center_and_circumference_angle(self):
        Update = False
        circles = self.logic.find_all_circles()
        for center in circles:
            points = self.logic.find_points_on_circle(center)
            for x, y, z in permutations(points, 3):
                if self.logic.cross(center, x, y) > 0: continue
                if all([self.logic.check_line((p, q)) for p, q in [(x, z), (y, z)]]):
                    center_angle = self.logic.find_arc_measure((center, x, y))[0]
                    Update = self.logic.defineAngle(x, z, y, 0.5 * center_angle) or Update
        return Update

    def func10_parallel_lines_theorem(self):
        Update = False
        parallels = self.logic.find_all_parallels()
        for line1, line2 in parallels:
            # If the line is represented by symbol, we should change it to two points.
            line1, line2 = self.logic.parseLine(line1), self.logic.parseLine(line2)

            # We want to figure out the order of the parallel lines.
            if self.logic.point_positions is not None:
                fdis = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
                dismax, idp, idq = 0, None, None
                for p, q in product(line1, line2):
                    nowdis = fdis(self.logic.point_positions[p], self.logic.point_positions[q])
                    if nowdis > dismax:
                        dismax = nowdis
                        idp, idq = p, q
                if line1.index(idp) == line2.index(idq):
                    line1 = line1[::-1]

            # Now start to use Parallel Theorems
            A = self.logic.find_all_points_on_line(line1)
            B = self.logic.find_all_points_on_line(line2)
            for p, q in product(A, B):
                if not self.logic.check_line((p, q)): continue
                angles = [(A[0], p, q), (A[-1], p, q), (B[0], q, p), (B[-1], q, p)]
                measures = [self.logic.find_angle_measure(x) for x in angles]
                # It is guaranteed that all the angle has at least one symbol.
                if A[0] != p and B[-1] != q:
                    Update = self.logic.angleEqual(angles[0], angles[3]) or Update
                if A[-1] != p and B[0] != q:
                    Update = self.logic.angleEqual(angles[1], angles[2]) or Update
                if A[0] != p and B[0] != q:
                    Update = self.logic.define_angle_measure(*angles[0], 180 - measures[2][0]) or Update
                if A[-1] != p and B[-1] != q:
                    Update = self.logic.define_angle_measure(*angles[1], 180 - measures[3][0]) or Update
        return Update

    def func11_flat_angle_theorem(self):
        # If point O lies on segment (A, B), then AOC + COB = 180.
        Update = False
        angles = self.logic.find_all_angle_measures()
        for angle in angles:
            if angle[3] != 180: continue
            points = set(self.logic.find_all_lines_for_point(angle[1])) - \
                     set(self.logic.find_all_points_on_line((angle[0], angle[2])))
            for point in points:
                val = self.logic.find_angle_measure((point, angle[1], angle[2]))[0]
                Update = self.logic.define_angle_measure(angle[0], angle[1], point, 180 - val) or Update
        return Update

    def func12_intersecting_chord_theorem(self):
        circles = self.logic.find_all_circles()
        for center in circles:
            points = self.logic.find_points_on_circle(center)
            for p1, p2, p3, p4 in permutations(points, 4):
                if self.logic.check_line((p1, p2)) and self.logic.check_line((p3, p4)):
                    intersection = set(self.logic.find_all_points_on_line((p1, p2))) & \
                                   set(self.logic.find_all_points_on_line((p3, p4)))
                    if len(intersection) != 1: continue
                    inter = intersection.pop()
                    l1, l2, l3, l4 = map(self.logic.find_line_with_length,
                                         [(p1, inter), (p2, inter), (p3, inter), (p4, inter)])
                    expr = l1[0] * l2[0] - l3[0] * l4[0]
                    if len(expr.free_symbols) <= 2:
                        self.equations.append(expr)

    def func13_polygon_interior_angles_theorem(self):
        # Equations in Quadrilateral
        Quadrilaterals = self.logic.find_all_quadrilaterals()
        for quad in Quadrilaterals:
            expr = sum(
                [self.logic.find_angle_measure((quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]))[0] for i in range(4)])
            self.equations.append(expr - 180 * (len(quad) - 2))

    def func14_similar_triangle_theorem(self):
        triangles = self.logic.find_all_triangles()
        comb = combinations(triangles, 2)
        for pair in comb:
            for tri1 in permutations(pair[0]):
                tri2 = pair[1]
                lines = self._generateLines(tri1) + self._generateLines(tri2)
                length = [self.logic.find_line_with_length(x) for x in lines]
                angles = [self.logic.find_angle_measure(x) for x in
                          self._generateAngles(tri1) + self._generateAngles(tri2)]
                s = self._same
                same_angle = [s(angles[0], angles[3]), s(angles[1], angles[4]), s(angles[2], angles[5])]
                if self._traingleSimilar(same_angle) or self.logic.check_similar_triangle(tri1, tri2):
                    for ch1, ch2 in combinations([0, 1, 2], 2):
                        if sum([hasNumber(x) for x in
                                [length[ch1], length[ch1 + 3], length[ch2], length[ch2 + 3]]]) >= 2:
                            expr = length[ch1][0] * length[ch2 + 3][0] - length[ch2][0] * length[ch1 + 3][0]
                            self.equations.append(expr)

    def func15_angle_bisector_theorem(self):
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for z in tri:
                x = tri[1] if z == tri[0] else tri[0]
                y = tri[1] if z == tri[2] else tri[2]
                points = self.logic.find_all_points_on_line((x, y))
                for m in points:
                    if m != x and m != y and self._same(self.logic.find_angle_measure((x, z, m)),
                                                        self.logic.find_angle_measure((y, z, m))):
                        s1, s2, t1, t2 = map(self.logic.find_line_with_length, ((x, z), (y, z), (x, m), (y, m)))
                        # s1 / t1 = s2 / t2
                        expr = s1[0] * t2[0] - t1[0] * s2[0]
                        self.equations.append(expr)






    def initSearch(self):
        self.logic.put_equal_into_symbol()
        self.logic.try_delete_unused_points()  # remove no-use points
        self.logic.init_all_uni_lines()  # initialize all the uni-lines (the line do not contain other lines)
        self.logic.find_hidden_polygons(3)  # find all triangles
        self.logic.find_hidden_polygons(4)  # find all quads
        self.logic.expand_angles()  # add the rest (hidden) angles into the graph and give each of them a symbol
        self.logic.set_angle_sum()  # resolve the relation among angles
        self.logic.set_line_sum()  # resolve the relation among lines (e.g, AB+BC = AC)
        # self.logic.set_arc_sum()  # resolve the relation among arcs

        if self.logic.debug:
            print(self.logic.find_all_angle_measures())
            print(self.logic.find_all_lines_with_length())
            print(self.logic.fine_all_arc_measures())
        self.can_search = True
        self.hasSolution = False

    def Search(self, target, order_list, round_or_step, upper_bound, enable_low_first):
        """
        This is the main search process.
        Args:
            target[0] = 'Value'
                target[1:] = ['x']: Find the value of variable x.
                target[1:] = ['A', 'B']: Find the length of line AB.
                target[1:] = ['A', 'B', 'C']: Find the measure of angle ABC.
                target[1:] = ['arc_measure', 'O', 'M', 'N']: Find the measure of arc OMN.
                target[1:] = ['arc_length', 'O', 'M', 'N']:  Find the length of arc OMN.
            target[0] = 'Area'
                target[1:] = ['A', 'B', 'C']: Find the area of triangle ABC.
                target[1:] = ['A', 'B', 'C', 'D']: Find the area of quad ABCD.
            target[0] = 'Perimeter'
                target[1:] = ['O']: Find the perimeter of circle O.
                target[1:] = ['A', 'B', 'C']: Find the perimeter of triangle ABC.
                target[1:] = ['A', 'B', 'C', 'D']: Find the perimeter of quad ABCD.
            target[0] = 'SinOf/CosOf/TanOf/CotOf/HalfOf/SquareOf/SqrtOf/RatioOf/Add/Mul/SumOf"'
                target[1:] = new_target
                For example, [SinOf, [value, 'A', 'B', 'C']] or [Add, ['value', 'A', 'B'], ['value', 'C', 'D']]
        Returns:
            (float) The value of the target.

        Note: if the goal is to find 'sin \\angle ABC', use this function to calc 'value ABC'
            and finally do the sine outside this function.
        """


        # try to get the answer before using theorems
        step_lst = []
        now_answer = self._getAnswer(target)
        if now_answer is not None:
            return now_answer, 0, step_lst

        if round_or_step:
            rounds = 0
            Update = True
            while rounds <= upper_bound and Update:
                rounds += 1
                Update = False
                self.equations = []
                for i in range(1, len(self.function_maps)+1):
                    if enable_low_first and i > self.interval: continue
                    step_lst.append(i)
                    changed = self.function_maps[i]()
                    if changed is not None and changed:
                        Update = True
                Update = Update or len(self.equations) > 0
                FindSolution = self.Solve_Equations()
                now_answer = self._getAnswer(target)
                if now_answer is not None:
                    return now_answer, rounds, step_lst
                if not enable_low_first or FindSolution: continue
                self.equations = []
                for i in range(self.interval+1, len(self.function_maps)+1):
                    step_lst.append(i)
                    self.function_maps[i]()
                Update = Update or len(self.equations) > 0
                self.Solve_Equations()
                now_answer = self._getAnswer(target)
                if now_answer is not None:
                    return now_answer, rounds, step_lst
            return None, rounds, step_lst

        else:
            # check order_lst
            func_ids = self.function_maps.keys()
            if isinstance(order_list, list) and len(order_list) > 0:
                result = all([order in func_ids for order in order_list])
                if not result:
                    order_list = []
            else:
                order_list = []

            # Run the predicting series.
            steps = 0
            for element in order_list:
                steps += 1
                step_lst.append(element)
                self.equations = []
                self.function_maps[element]()
                self.Solve_Equations()
                now_answer = self._getAnswer(target)
                if now_answer is not None:
                    return now_answer, steps, step_lst

            Update = True
            while Update and steps < upper_bound:
                # print (self.logic.find_all_angle_measures())
                # print (self.logic.find_all_lines_with_length())
                # print (self.logic.variables)
                Update = False
                FindSolution = False
                for i in range(1, len(self.function_maps)+1):
                    if enable_low_first and i > self.interval:
                        continue
                    steps += 1
                    step_lst.append(i)
                    self.equations = []
                    changed = self.function_maps[i]()
                    if changed is not None and not changed or changed is None and len(self.equations) == 0:
                        continue
                    Update = True
                    FindSolution = self.Solve_Equations() or FindSolution
                    now_answer = self._getAnswer(target)
                    if now_answer is not None:
                        return now_answer, steps, step_lst

                if not enable_low_first or FindSolution: continue
                for i in range(self.interval+1, len(self.function_maps)+1):
                    steps += 1
                    step_lst.append(i)
                    self.equations = []
                    self.function_maps[i]()
                    if len(self.equations) == 0: continue
                    Update = True
                    self.Solve_Equations()
                    now_answer = self._getAnswer(target)
                    if now_answer is not None:
                        return now_answer, steps, step_lst
            return None, steps, step_lst

    def Find_relations(self):
        update_lst = []
        for i in range(1, len(self.function_maps)+1):
            update = self.function_maps[i]()
            if update:
                print(self.function_maps[i].__name__)
            
    def beamSearch(self):
        pass
        
        
def parse_logic_forms(diagram_parser):
    parser = LogicParser(ExtendedDefinition(debug=False))
    parser.logic.point_positions = diagram_parser['point_positions']

    isLetter = lambda ch: ch.upper() and len(ch) == 1
    parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])

    lines = diagram_parser['line_instances']  # ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    for line in lines:
        line = line.strip()
        if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
            parser.logic.define_line(line[0], line[1])

    circles = diagram_parser['circle_instances']  # ['O']
    for point in circles:
        parser.logic.define_circle(point)

    # Parse diagram logic forms
    logic_forms = diagram_parser['diagram_logic_forms']
    logic_forms = sorted(logic_forms, key=lambda x: x.find("Perpendicular") != -1)  # put 'Perpendicular' to the end

    for logic_form in logic_forms:
        if logic_form.strip() != "":
            try:
                parse_tree = parser.parse(logic_form) # ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
                parser.dfsParseTree(parse_tree)
            except Exception as e:
                print(logic_form)
                
    solver = LogicSolver(parser.logic)
    solver.logic.find_hidden_polygons(3)  # find all triangles
    solver.logic.find_hidden_polygons(4)
    # data = solver.Find_relations()
    print(solver.logic.Triangle.facts)
    print(solver.logic.Quadrilateral.facts)
    print(solver.logic.Equal.facts)
    return {
        "Triangle": list(solver.logic.Triangle.facts),
        "Quad": list(solver.logic.Quadrilateral.facts),
    }
    return data