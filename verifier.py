import json
import os
import itertools
from copy import deepcopy
from formalgeo.solver import Interactor
from formalgeo.parse.basic import parse_geo_predicate, parse_equal_predicate, get_equation_from_tree

from utils.tools import read_json
from utils.symbolic import parse_clause, build_point_map, replace_points_for_clause
from utils.formulate import clause_to_nature_language
from sympy import simplify

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
    
    def verify_existance(self, conditions):
        for clause in conditions:
            name = clause.split('(')[0]
            if name in self.predicate_names:
                items = self.solver.problem.condition.get_items_by_predicate(name)
                _, para, _ = parse_geo_predicate(clause)
                if not self.check_paras(name, para, items):
                    statement = clause_to_nature_language(
                        [clause], 
                        natural_template=self.natural_template
                    )[0]
                    msg = f"Relation {statement} doesn't exist."
                    return False, msg
                
            elif name in ['Value', 'Equal']:
                clause = clause.replace('Value', 'Equal')
                eq_tree, attrs = parse_equal_predicate(clause)
                eq_expr = get_equation_from_tree(self.solver.problem, eq_tree[1])
                equations = [item[1] for item in self.solver.problem.condition.items if item[0] == 'Equation']
                if not self.check_equations(eq_expr, equations):
                    msg = f"Relation {clause} doesn't exist."
                    return False, msg
                
            elif name == 'Equation':
                eq = clause.replace('Equation(', '').replace(')', '')
                equations = [item[1] for item in self.solver.problem.condition.items if item[0] == 'Equation']
                if not self.check_equations(eq, equations):
                    msg = f"There's no equation {eq}"
                    return False, msg
            
        return True, None
    
    def verify_theorem(self, theorem, conditions, conclusions):
        if theorem is None:
            # extend all conclusions first
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
            for cclsn in conclusions:
                cclsn = cclsn.replace('Value', 'Equal')
                if cclsn not in conclusions_total:
                    msg = f"Conclusion {clause} can't be inferred."
                    return False, msg
                
        else:
            # apply solver directly
            update = self.solver.apply_theorem_by_name(theorem)
            if not update:
                vecify_msg = f"Theorem {theorem} can't be applied."
                return False, vecify_msg
            
        return True, None
    
    def verify(self, solution_dict):
        for step, info in solution_dict.items():
            theorem = info['theorem']
            conditions = info['condition']
            conclusions = info['conclusion']
            
            # check condition existance
            verify_flag, vecify_msg = self.verify_existance(conditions)
            if not verify_flag:
                return False, vecify_msg
            
            # check application of theorem
            verify_flag, vecify_msg = self.verify_theorem(theorem, conditions, conclusions)
            if not verify_flag:
                return False, vecify_msg
                
            # check conclustion existance
            verify_flag, vecify_msg = self.verify_existance(conclusions)
            if not verify_flag:
                return False, vecify_msg
            
        return True, None
    
    
if __name__ == '__main__':
    test_solution_path = 'geo_synth_2/geosynth_ENT_1_REL_1/annotations/test.json'
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
