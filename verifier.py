import json
import os
import itertools
from copy import deepcopy
# from formalgeo.solver import Interactor
from formalgeo.parse.basic import parse_geo_predicate, parse_equal_predicate, get_equation_from_tree
from solver import Interactor
from utils.tools import read_json
from utils.symbolic import parse_clause, build_point_map, replace_points_for_clause
from utils.formulate import clause_to_nature_language
from sympy import simplify, solve, Eq, sin, cos, tan, rad

def rotate_combinations(lst):
    n = len(lst)
    for i in range(n):
        rotated = lst[i:] + lst[:i]
        yield rotated

class Verifier():
    def __init__(self,
                 predicate_GDL,
                 theorem_GDL,
                 t_info,
                 natural_template,
                 debug=False
                 ):
        self.predicate_GDL = predicate_GDL
        self.theorem_GDL = theorem_GDL
        self.t_info = t_info
        self.natural_template = natural_template
        self.debug = debug
        
        # build map: predicate to extend conclusions 
        self.predicate_to_extend = {}
        for template, info in self.predicate_GDL['Entity'].items():
            name = template.split('(')[0]
            self.predicate_to_extend[name] = {
                'template': template,
                'extend': info['extend']
            }
        for template, info in self.predicate_GDL['Relation'].items():
            name = template.split('(')[0]
            self.predicate_to_extend[name] = {
                'template': template,
                'extend': info['extend']
            }
        
    def init_problem(self, text_cdls, constr_cdls, image_cdls):
        self.solver = Interactor(
            self.predicate_GDL,
            self.theorem_GDL,
            self.t_info,
            debug=self.debug
        )
        self.problem_CDL = {
            "problem_id": 0,
            "construction_cdl": constr_cdls,
            "text_cdl": text_cdls,
            "image_cdl": image_cdls,
            "goal_cdl": f"Value(LengthOfLine(AB))",
            "problem_answer": "0",
        }
        self.solver.load_problem(self.problem_CDL)
        self.predicate_names = list(self.solver.problem.condition.items_group.keys())
        self.predicate_names.remove('Equation')
        
    def check_equations(self, eq_expr, equations):
        for eq_i in equations:
            res_expr = simplify(f"({str(eq_expr)}) - ({str(eq_i)})")
            if res_expr == 0:
                return True
            
        return False
    
    def check_paras(self, name, para, items):
        if type(para) != tuple:
            para = tuple(para)
        if name in ['Line', 'Shape', 'Polygon', 'Collinear', 'Triangle']:
            # No location restriction
            para_permutations = list(itertools.permutations(para))
            if any([para_i in items for para_i in para_permutations]):
                return True
        elif name in ['EquilateralTriangle', 'Parallelogram', 'Kite', 'Rhombus', 'Rectangle', 'Square', 'Trapezoid', 'IsoscelesTrapezoid']:
            # Can be rotation
            para_rotations = rotate_combinations(para)
            if any([para_i in items for para_i in para_rotations]):
                return True
        else:  
            # Other cases
            if para in items:
                return True

        return False
    
    def clause_to_expr_str(self, clause):
        name = clause.split('(')[0]
        assert name in ['Value', 'Equal', 'Equation']
        if name in ['Value', 'Equal']:
            clause = clause.replace('Value', 'Equal')
            eq_tree, attrs = parse_equal_predicate(clause)
            eq_expr = get_equation_from_tree(self.solver.problem, eq_tree[1])
        elif name == 'Equation':
            eq_expr = clause.replace('Equation(', '').rstrip(')')
            
        return str(eq_expr)
    
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
            # extend all conclusions first (recursion)
            conclusions_total = []
            for clause in conditions:
                name = clause.split('(')[0]
                if name in self.predicate_to_extend:
                    clause_template = self.predicate_to_extend[name]['template']
                    extends_template = self.predicate_to_extend[name]['extend']
                    # template -> clause
                    p_map = build_point_map(clause, clause_template)
                    for extend_temp in extends_template:
                        extend_clause = replace_points_for_clause(extend_temp, p_map)
                        conclusions_total.append(extend_clause)
                else:
                    raise KeyError(name)
            # then check conclusion existance
            for cclsn, statement in zip(conclusions, cclsn_statements):
                cclsn = cclsn.replace('Value', 'Equal')
                if cclsn not in conclusions_total:
                    msg = f"Conclusion {statement} can't be inferred through conditions:\n{','.join(cdtn_statements)}."
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
                eq_str = '\n .'.join(cdtn_statements)
                verify_msg = f"Equations group:\n{eq_str}\ncan't be solved."
                return False, verify_msg
            
            else:
                if isinstance(solution, list):
                    solution = solution[0]
                result_eqs = [sym - value for sym, value in solution.items()]
                # check conclusion in solved results
                for expr, statement in zip(cclsn_expr_lst, cclsn_statements):
                    if not self.check_equations(expr, result_eqs):
                        condition_eq_str = '\n.'.join(cdtn_statements)
                        verify_msg = f"Conclusion {statement} can't be solved from equations group:\n{condition_eq_str}"
                        return False, verify_msg
                    
        elif theorem not in self.solver.parsed_theorem_GDL:
            added_theorems = ['cos_of_angle', 'sin_of_angle', 'tan_of_angle']
            # 需要往solver / interactor中添加额外的规则（在ForwardSearcher中）
            # 不然有的题解不出来
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
                    # l2 = cos(a) * l1
                    # l3 = sin(a) * l1
                    # l3 = tan(a) * l2
                    l1 = (angle_sp[0], angle_sp[1])
                    l2 = (angle_sp[1], angle_sp[2])
                    l3 = (angle_sp[2], angle_sp[0])
                    if angle_sp[0] == angle_90[1]:
                        l1, l2 = l2, l1
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
                        verify_msg = f"Conclusion {cclsn_statements[0]} can't be solved from {theorem}, {', '.join(cdtn_statements)}"
                        return False, verify_msg
            else:
                raise KeyError(theorem)
        else:
            # apply solver directly
            update = self.solver.apply_theorem_by_name(theorem)
            if not update:
                verify_msg = f"Theorem {theorem} can't be applied."
                return False, verify_msg
            
        return True, None
    
    def verify(self, solution_dict):
        for step, info in solution_dict.items():
            theorem = info['theorem']
            conditions = info['condition']
            conclusions = info['conclusion']
            
            # check condition existance
            verify_flag, verify_msg = self.verify_existance(conditions)
            if not verify_flag:
                return False, verify_msg
            
            # check application of theorem
            verify_flag, verify_msg = self.verify_theorem(theorem, conditions, conclusions)
            if not verify_flag:
                return False, verify_msg
                
            # check conclustion existance
            verify_flag, verify_msg = self.verify_existance(conclusions)
            if not verify_flag:
                return False, verify_msg
            
        return True, None
    
    
if __name__ == '__main__':
    test_solution_path = 'geo_synth_2/geosynth_ENT_1_REL_1/annotations/test_1.json'
    test_data = read_json(test_solution_path)
    test_solution_dict = test_data['llm_info']['solution_dict']
    
    predicate_GDL_search = read_json('json/predicate_GDL_for_search.json')
    theorem_GDL_search = read_json('json/theorem_GDL_for_search.json')
    t_info = read_json("json/t_info_new.json")
    natural_template = read_json("json/predicates_to_nature_language.json")
    
    verifier = Verifier(
        predicate_GDL=predicate_GDL_search,
        theorem_GDL=theorem_GDL_search,
        t_info=t_info,
        natural_template=natural_template,
        debug=True
    )
    verifier.init_problem(
        text_cdls=test_data['text_cdl'],
        constr_cdls=test_data['construction_cdl'],
        image_cdls=test_data['image_cdl'],
    )
    res, msg = verifier.verify(test_solution_dict)
    print(res)
    print(msg)
