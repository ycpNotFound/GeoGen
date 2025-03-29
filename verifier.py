import itertools
import json
import os
from copy import deepcopy

from sympy import Eq, cos, rad, simplify, sin, solve, tan

# from formalgeo.solver import Interactor
from formalgeo_v2.parse.basic import (get_equation_from_tree,
                                   parse_equal_predicate, parse_geo_predicate)
from solver import Interactor
from utils.formulate import clause_to_nature_language
from utils.symbolic import (build_point_map, parse_clause,
                            replace_points_for_clause)
from utils.tools import read_json
from tqdm import tqdm

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
                 strict=False,
                 debug=False
                 ):
        self.predicate_GDL = predicate_GDL
        self.theorem_GDL = theorem_GDL
        self.t_info = t_info
        self.natural_template = natural_template
        self.strict = strict
        self.debug = debug
        
        # build map: predicate to extend conclusions 
        self.predicate_to_extend_map = {}
        for template, info in self.predicate_GDL['Entity'].items():
            name = template.split('(')[0]
            self.predicate_to_extend_map[name] = {
                'template': template,
                'extend': info['extend'],
                'multi': info['multi']
            }
        for template, info in self.predicate_GDL['Relation'].items():
            name = template.split('(')[0]
            self.predicate_to_extend_map[name] = {
                'template': template,
                'extend': info['extend'],
                'multi': info['multi']
            }
        # build attr_dict: ll_ -> LengthOfLine
        self.attr_dict = self.build_attr_dict()

    def build_attr_dict(self):
        attr_dict = {}
        for k, v in self.predicate_GDL['Attribution'].items():
            sym = v['sym'] + '_'
            predicate = k.split('(')[0]
            attr_dict[sym] = predicate

        return attr_dict
        
    def init_problem(self, text_cdls, constr_cdls, image_cdls, p_pos=None, lines=None):
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
            "goal_cdl": f"Value(q)",
            "problem_answer": "0",
            "point_positions": p_pos if p_pos else [],
            "line_instances": lines if lines else [],
        }
        self.solver.load_problem(self.problem_CDL, solve_eq=self.strict)
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
            eq_expr = clause.replace('Equation(', '')
            eq_expr = eq_expr[:-1]
            
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
                    new_clause_template = f"{name}({para_str})"
                    # two mapping: ori_temp -> new_temp -> new_clause
                    new_clause = replace_points_for_clause(new_clause_template, p_map)
                    conclusions_total += self.extend_one_condition(new_clause)
                    
            elif name in ['Equal', 'Value', 'Equation']:
                return []
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
    
    def verify_strict(self, solution_dict):
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
    
    def verify_premise_of_conditions(self, conditions):
        # statements = clause_to_nature_language(
        #     conditions, 
        #     natural_template=self.natural_template
        # )
        for clause in conditions:
            name = clause.split('(')[0]
            if name in self.predicate_names:
                items = self.solver.problem.condition.get_items_by_predicate(name)
                _, para, _ = parse_geo_predicate(clause)
                para = tuple(para)
                # if has existed in conditions
                if self.check_paras(name, para, items):
                    return True, None
                
                # check the premise of clause
                if not self.solver.problem.ee_check(name, para):
                    w_msg = "EE check not passed: [{}, {}]".format(name, items)
                    return False, w_msg
                
                if not self.solver.problem.fv_check(name, para):  # fv check
                    w_msg = "FV check not passed: [{}, {}]".format(name, items)
                    return False, w_msg
                
                # treated as True if premises of it exist, and add the clause into conditions
                self.solver.problem.add(
                    name, para, (-1, ), ('prerequisite', None, None), skip_check=True
                )
                return True, None
                
            elif name in ['Value', 'Equal', 'Equation']:
                eq_str = self.clause_to_expr_str(clause)
                symbols = [str(s) for s in list(simplify(eq_str).free_symbols)]
                for sym in symbols:
                    if not self.verify_symbol_existance(sym):
                        msg = f"{sym} is not in existed symbols."
                        return False, msg


                return True, None
                
            else:
                raise KeyError(name)
            
        return True, None
        
    def verify_symbol_existance(self, sym_str):
        if str(sym_str) in [str(s) for s in self.solver.problem.condition.value_of_sym]:
            return True
        for sym_prefix, predicate in self.attr_dict.items():
            if sym_prefix in sym_str:
                para = tuple(sym_str.split(sym_prefix)[-1].upper())

                res = self.solver.problem.ee_check(predicate, para)
                if not res:
                    return False
                else:
                    self.solver.problem.get_sym_of_attr(predicate, para)
        return True

    def verify_premise(self, solution_dict):
        for step, info in solution_dict.items():
            theorem = info['theorem']
            conditions = info['condition']
            conclusions = info['conclusion']
            
            # check condition existance
            verify_flag, verify_msg = self.verify_premise_of_conditions(conditions)
            if not verify_flag:
                return False, verify_msg
        
                
            # check conclustion existance
            verify_flag, verify_msg = self.verify_premise_of_conditions(conclusions)
            if not verify_flag:
                return False, verify_msg
            
        return True, None
    
    def verify(self, solution_dict):
        if self.strict:
            return self.verify_strict(solution_dict)
        else:
            return self.verify_premise(solution_dict)

def test_verify():
    test_solution_dir = 'datasets/pgps/pgps_train_search'
    symbolic_info = read_json('D:/Desktop/资源/几何答题/UniAll/total_geo_expand_train_symbolic.json')
    symbolic_pgps_info = {
        v['fgo_key']: v for k, v in symbolic_info.items()
        if 'pgps' in k
    }
    predicate_GDL_search = read_json('json/predicate_GDL_for_search.json')
    theorem_GDL_search = read_json('json/theorem_GDL_for_search.json')
    t_info = read_json("json/t_info_new.json")
    natural_template = read_json("json/predicates_to_nature_language.json")

    cnt = 0
    files = os.listdir(test_solution_dir)
    for f in tqdm(files):
        cnt += 1
        
        fgo_key = f.split('.')[0]
        if fgo_key not in symbolic_pgps_info:
            continue
        # if fgo_key != 'img_1001_1':
        #     continue
        test_data = read_json(f"{test_solution_dir}/{f}")
        test_symbolic_info = symbolic_pgps_info[fgo_key]
        test_solution_dict = test_data['llm_info']['solution_dict']

        strict = False
        verifier = Verifier(
            predicate_GDL=predicate_GDL_search,
            theorem_GDL=theorem_GDL_search,
            t_info=t_info,
            natural_template=natural_template,
            debug=True,
            strict=strict
        )
        verifier.init_problem(
            text_cdls=test_data['text_cdl'],
            constr_cdls=test_data['construction_cdl'],
            image_cdls=test_data['image_cdl'],
            p_pos=test_symbolic_info['p_pos'] if 'p_pos' in test_symbolic_info else None,
            lines=test_symbolic_info['lines'] if 'lines' in test_symbolic_info else None,
        )
        res, msg = verifier.verify(test_solution_dict)
        # print(res)
        # print(msg)
    
if __name__ == '__main__':
    test_verify()
