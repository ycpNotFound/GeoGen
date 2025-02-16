import itertools
import random
import re
import string
from copy import deepcopy
from tqdm import tqdm
import networkx as nx
from sympy import sin, cos, tan, sqrt, pi, Symbol
from formalgeo.data import DatasetLoader
from formalgeo.parse import (parse_predicate_gdl, parse_problem_cdl,
                             parse_theorem_gdl)
from utils.preset import PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2
from utils.symbolic import (find_target_for_construct, get_content,
                            get_points_mapping, get_points_num,
                            get_predicate_name, parse_clause, replace_points)
from utils.tools import append_lst, setup_seed
from formalgeo.parse.parse_cdl import parse_one_theorem

class ClauseGenerator():
    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate_GDL = predicate_GDL
        self.predicate_ent_names = {s.split('(')[0]: s for s in predicate_GDL['Entity']}
        self.predicate_rel_names = {s.split('(')[0]: s for s in predicate_GDL['Relation']}
        self.theorem_GDL = theorem_GDL
        self.theorem_names = list(theorem_GDL.keys())
        
        self.all_letters = list(string.ascii_lowercase)
        self.points = []
        self.lines = []
        self.circles = []
        self.polygons = []
        self.polygons_used = []
        
        self.constraints = []
        self.constraints_base = []
        
        self.points_on_circle = {}
        self.points_with_constraints = {}

    
    @property
    def p_num(self):
        return len(self.points)
    
    @property
    def l_num(self):
        return len(self.lines)
    
    @property
    def states(self):
        return {
            "points": self.points,
            "lines": self.lines,
            "circles": self.circles,
            "polygons": self.polygons,
            "constraints": self.constraints,
            "constraints_base": self.constraints_base,
            "points_on_circle": self.points_on_circle
        }
        
    def empty_states(self):
        self.points = []
        self.lines = []
        self.circles = []
        self.polygons = []
        self.constraints = []
        self.constraints_base = []
        
        self.points_on_circle = {}
        self.points_with_constraints = {}
    
    
    def print_states(self):
        p_str = ', '.join(self.points)
        l_str = ', '.join([''.join(l)  for l in self.lines])
        c_str = ', '.join(self.circles)
        
        print(f"Points: {p_str}")
        print(f"Lines: {l_str}")
        print(f"Circles: {c_str}")
        print('Constraints: ')
        for c in self.constraints:
            if c not in self.constraints_base:
                print(c)
            
    def get_extend(self, clause):
        name = get_predicate_name(clause)
        if name in self.predicate_ent_names:
            template_clause = self.predicate_ent_names[name]
            template_info = deepcopy(self.predicate_GDL['Entity'][template_clause])
        elif name in self.predicate_rel_names:
            template_clause = self.predicate_rel_names[name]
            template_info = deepcopy(self.predicate_GDL['Relation'][template_clause])
        elif name in ['Equal']:
            return [clause]
        else:
            print(f'Error: {name} not in GDLs')
            raise KeyError(name)
        
        if len(template_info['extend']) == 0:
            return [clause]
        mapping = get_points_mapping(clause, template_clause)
        _, info = replace_points(clause, template_info, None, mapping)
        
        constraints_i = []
        for item in info['extend']:
            constraints_i += self.get_extend(item)
            
        return constraints_i
            
    def get_all_extend(self, clauses):
        constraints = []
        for clause in clauses:
            try:
                constraints += self.get_extend(clause)
            except RecursionError as e:
                print(clauses)
                raise e
        return constraints
    
    def update_constraints_for_points(self, clauses, targets_in_clauses=None):
        '''record / add constraints (clauses) for each point'''
        for p in self.points:
            if p not in self.points_with_constraints:
                self.points_with_constraints[p] = []
                
        if targets_in_clauses is None:
            targets_in_clauses = [find_target_for_construct(c) for c in clauses] 
        for targets, clause in zip(targets_in_clauses, clauses):
            for p in targets:
                self.points_with_constraints[p] = append_lst(
                    self.points_with_constraints[p], [clause])
                # Similar和Congruent虽然只有1条约束，按2条约束处理
                if 'Similar' in clause or 'Congruent' in clause:
                    self.points_with_constraints[p].append('')
                    
                # 出现Polygon(ABCD)也按2条约束处理，直接固定住
                if 'Polygon' in clause:
                    _, items = parse_clause(clause)
                    if len(items[0]) == 4:
                        self.points_with_constraints[p].append('')
        return
            
    def find_unconstrained_points(self, n=None, ignore_ps=None):
        unconstr_ps = []
        for p, constrs in self.points_with_constraints.items():
            # do not return point a, b, c, as it's used for initialization
            if p in ['a', 'b', 'c']:
                continue
            if ignore_ps is not None and p in ignore_ps:
                continue
            # if p is circle center defined by 'Cocircular' with >3 points, 
            # it's constrained
            if p in self.circles:
                if len(self.points_on_circle[p]) <= 2:
                    unconstr_ps.append(p)

            else:
                if len(constrs) < 2:
                    unconstr_ps.append(p)
                
        if len(unconstr_ps) == 0 and n is not None:
            unconstr_ps = self.add_new_points(1)
                
        if n is not None: # create new if not enough
            if n > len(unconstr_ps):
                new_ps = self.add_new_points(n - len(unconstr_ps))
                unconstr_ps += new_ps
            else:
                unconstr_ps = random.sample(unconstr_ps, n)
            
        return unconstr_ps
        
            
    def generate_clauses_from_predicates(
        self, 
        predicates_base, 
        predicates_rel, 
        n_more_lines
    ):
        # define base entity (Square, Rectangle / Similar, Congruent) 
        self.predicates_base = predicates_base
        self.predicates_rel = predicates_rel
        constr_cdls = [] 
        text_cdls = [] 
        for name in predicates_base:
            pred_type = "Entity" if name in PREDICATES_ENT else "Relation"
            constr_cdl, text_cdl = self.define_base(name, pred_type=pred_type)
            constr_cdls += constr_cdl
            text_cdls += text_cdl
            
        # define relation predicates (Midpoint, Parallel ..)
        for name in predicates_rel:
            constr_cdl, text_cdl = self.define_relation(name)
            constr_cdls += constr_cdl
            text_cdls += text_cdl
            self.find_collinear_from_para_perp()
            self.find_triangles()
            
        
        constr_cdl = self.add_more_lines(text_cdls, n_more_lines)
        self.find_triangles()
        constr_cdls += constr_cdl
        return constr_cdls, text_cdls

    
    def define_base_v2(self, clause_base):
        predicate, items = parse_clause(clause_base)
        pred_info = deepcopy(self.predicate_GDL["Entity"][predicate])
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        text_cdl = [clause_base]
        # add constraints: ab \\parallel cd ..
        # items -> points
        self.polygons = append_lst(self.polygons, [tuple(points)])
        all_extend = self.get_all_extend(pred_info['extend'])
        if len(all_extend) == 0:
            all_extend = [clause_base]
        self.constraints += all_extend
        self.constraints_base += all_extend
        self.update_constraints_for_points(all_extend)
        return constr_cdl, text_cdl
    
    def define_relation_v2(self, clause_relation):
        constr_cdls = []
        text_cdls = []
        pred_name, items = parse_clause(clause_relation)
        predicate = self.predicate_rel_names[pred_name]
        # IsMidpointOfLine(M,AB)
        pred_info = deepcopy(self.predicate_GDL["Relation"][predicate])

        # get construct and text cdls
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        constr_cdls += constr_cdl
        text_cdls += [clause_relation]
        # add constraints
        all_extend = self.get_all_extend(pred_info['extend'])
        # add collinear / cocircular / polygon (> 4) to all_extend (have algebra relation)
        for cdl in constr_cdls:
            if 'Collinear' in cdl or 'Cocircular' in cdl:
                all_extend.append(cdl)
            if 'Polygon' in cdl:
                _, items = parse_clause(cdl)
                if len(items[0]) > 3:
                    all_extend.append(cdl)
        if len(all_extend) == 0:
            all_extend = [clause_relation]
        self.constraints += all_extend
        self.update_constraints_for_points(all_extend)
        return constr_cdls, text_cdls
    
    def define_clauses(self, clauses, given_map=None):
        # build point map
        p_map = {}
        if given_map is not None:
            p_map.update(given_map)
            
        points = get_para_from_clauses(clauses)
        for p in points:
            if p in p_map:
                continue
            new_p = self.add_new_points(1)[0]
            p_map[p] = new_p
        
        new_clauses = []
        for clause in clauses:
            name = clause.split('(')[0]
            # save new clause
            new_clause, _ = replace_points(clause, mapping=p_map)
            new_clauses.append(new_clause)
            # define relation
            if name == 'Equal':
                continue # do not need to define
            elif name in ["Point", "Line", "Arc", "Angle", "Circle", "Polygon", "Collinear", "Cocircular"]:
                self.define_construct_clauses([new_clause])
            elif name in PREDICATES_ENT:
                self.define_base_v2(new_clause)
            elif name in PREDICATES_REL:
                self.define_relation_v2(new_clause)
            else:
                raise KeyError(name)
            
        
        return new_clauses, p_map
    
    def generate_clauses_from_theorems(
        self,
        theorems
    ):
        # theorem chain:
        # t: theorem, c: condition, r: result / conclusion
        # apply t1: c0 -> r0
        # apply t2: c1 + r0 -> r1
        # apply t3: c2 + r1 -> r2
        # ......
        # 1. define points for c0, r0
        # 2. define points for c1, r1, in constraint: points in r0 the same.
        # 3. define points for c2, r2, in constraint: points in r1 the same.
        # ......
        self.theorems = theorems
        t_premises = [self.theorem_GDL[t]["1"]["premise"] for t in theorems]
        t_conclusions = [self.theorem_GDL[t]["1"]["conclusion"] for t in theorems]

        # initial conditions
        c0 = parse_premise_str(t_premises[0])
        
        r0 = t_conclusions[0][0]
        r0_name = process_equal_clause(r0) if 'Equal' in r0 else r0.split('(')[0]
        
        # define initial conditions (and the first conclusion)
        c0_new, p_map = self.define_clauses(c0)
        r0_new, p_map = self.define_clauses([r0], given_map=p_map)
        r0_new = r0_new[0]
        # additional init conditions
        c_i, r_i = c0, r0
        r_i_name = r0_name
        for i in range(1, len(theorems)):
            premise_clauses_i = parse_premise_str(t_premises[i])
            premise_names_i = [
                clause.split('(')[0] if 'Equal' not in clause
                else process_equal_clause(clause) 
                for clause in premise_clauses_i
            ]
            # build point map:
            # if conclusion is 'NAME(O,A)' -> 'NAME(a,b)'
            # premise next is 'NAME(P,ABC)', need to split into:
            # 'NAME(P,AB)' + 'NAME(P,C)', C(in c) -> A(in r) -> b
            
            # filter intermediate conclusion and append
            c_i = []
            for clause, name in zip(premise_clauses_i, premise_names_i):
                if name == r_i_name:
                    continue
                c_i.append(clause)
            
            # define c_i: condition_i
            self.define_clauses(c_i)
            
            # define r_i: results_i
            r_i = self.define_clauses(t_conclusions[i])[0]
            r_i_name = process_equal_clause(r_i) if 'Equal' in r_i else r_i.split('(')[0]

        
        return 
    
    def add_more_lines(self, text_cdls, n_more_lines):
        # 一些定制化的操作
        constr_cdls = []
        for text_cdl in text_cdls:
            predicate, items = parse_clause(text_cdl)
            if 'IsIncenterOfTriangle' == predicate:
                center, points = items
                for p in points:
                    self.add_new_line([center, p])
            if 'IsCentroidOfTriangle' == predicate or \
                'IsOrthocenterOfTriangle' == predicate: # "IsCentroidOfTriangle(O,ABC)"
                center, points = items
                p1 = random.choice(points)
                p2, p3 = sorted(list(set(points) - set([p1])))
                mid_p = self.add_new_points(1)[0]
                self.add_new_line([p1, center, mid_p])
                self.add_new_line([p2, mid_p, p3])
                constr_cdls += [
                    f"Collinear({''.join([p1, center, mid_p])})",
                    f"Collinear({''.join([p2, mid_p, p3])})",
                ]
            if 'IsTangentOfCircle' == predicate: # "IsTangentOfCircle(PA,O)"
                line, circle = items
                self.add_new_line([circle, line[1]])
            if 'IsMidpointOfArc' == predicate: # "IsMidpointOfArc(M,OAB)"
                # create line OM, OM and AB intersect -> OCM and ACB
                mid_p, points = items
                circle, p1, p2 = points
                new_p = self.add_new_points(1)[0]
                self.add_new_line([circle, new_p, mid_p])
                self.add_new_line([p1, new_p, p2])
                constr_cdls += [
                    f"Collinear({''.join([circle, new_p, mid_p])})",
                    f"Collinear({''.join([p1, new_p, p2])})",
                ]
                
        
        # 额外增加一些线段
        if n_more_lines == 0:
            return constr_cdls
        possible_lines = list(itertools.combinations(self.points, 2))
        possible_new_lines = []
        for line_1 in possible_lines:
            exist_flag = False
            for line_2 in self.lines:
                if all([p in line_2 for p in line_1]):
                    # line_1 \in line_2
                    exist_flag = True
                    break
            if not exist_flag:
                possible_new_lines.append(line_1)
        if len(possible_new_lines) >= n_more_lines:
            possible_new_lines_n = random.sample(possible_new_lines, k=n_more_lines)
            for l in possible_new_lines_n:
                self.add_new_line(l)
            
        return constr_cdls
        
    
    def find_triangles(self):
        '''find possible triangles'''
        possible_triangles = list(itertools.combinations(self.points, 3))
        triangles = []
        for points in possible_triangles:
            if points in self.polygons:
                continue
            
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
                if all([p in l for p in points]):
                    collinear = True
            
            if l1_exist and l2_exist and l3_exist and not collinear:
                triangles.append(points)
                
        self.polygons = append_lst(self.polygons, triangles)
        
    def find_collinear_from_para_perp(self):
        '''
            AD \parallel BC, BE \parallel AD => E B C collinear
            AB \perp BC, BE \perp BC => E B C collinear
            得到的collinear关系是推论，不参与constraint的计算
        '''
        
        # Parallel lines
        parallel_clauses = [c for c in self.constraints if 'ParallelBetweenLine' in c]
        G_para = nx.Graph()
        for clause in parallel_clauses:
            _, items = parse_clause(clause)
            line_1, line_2 = items
            line_1, line_2 = tuple(sorted(line_1)), tuple(sorted(line_2))
            G_para.add_edge(line_1, line_2)
            
        parallel_groups = [list(group) for group in nx.connected_components(G_para)]
        for group in parallel_groups:
            if len(group) >= 3:
                line_comb = list(itertools.combinations(group, 3))
                for comb in line_comb:
                    l1, l2, l3 = comb
                    if len(set(l1) & set(l2)) == 1:
                        collinear_1, collinear_2 = l1, l2
                    elif len(set(l2) & set(l3)) == 1:
                        collinear_1, collinear_2 = l2, l3
                    elif len(set(l1) & set(l3)) == 1:
                        collinear_1, collinear_2 = l1, l3
                    else:
                        continue
                    same_p = list(set(collinear_1) & set(collinear_2))[0]
                    p1 = list(set(collinear_1) - set([same_p]))[0]
                    p2 = list(set(collinear_2) - set([same_p]))[0]
                    self.add_new_line([min(p1, p2), same_p, max(p1, p2)])
        
        
        # Perp lines
        perp_clauses = [c for c in self.constraints if 'PerpendicularBetweenLine' in c]    
        perp_dict = {}
        for clause in perp_clauses:
            _, items = parse_clause(clause)
            line_1, line_2 = items
            line_1, line_2 = tuple(sorted(line_1)), tuple(sorted(line_2))
            if line_1 not in perp_dict:
                perp_dict[line_1] = [line_2]
            else:
                perp_dict[line_1].append(line_2)
                
            if line_2 not in perp_dict:
                perp_dict[line_2] = [line_1]
            else:
                perp_dict[line_2].append(line_1)
        
        for line, perp_group in perp_dict.items():
            if len(perp_group) >= 2:
                line_comb = list(itertools.combinations(group, 2))
                for comb in line_comb:
                    l1, l2 = comb
                    if len(set(l1) & set(l2)) == 1:   
                        same_p = list(set(l1) & set(l2))[0]
                        p1 = list(set(l1) - set([same_p]))[0]
                        p2 = list(set(l2) - set([same_p]))[0]
                        self.add_new_line([min(p1, p2), same_p, max(p1, p2)])
                        
    
    def define_base(self, pred_name, pred_type, ):
        '''define base entity (Square, Rectangle)'''
        assert pred_type in ["Entity", "Relation"]
        if pred_type == "Entity":
            predicate = self.predicate_ent_names[pred_name]
        else:
            predicate = self.predicate_rel_names[pred_name]
        # Predicate(ABC..)
        pred_info = deepcopy(self.predicate_GDL[pred_type][predicate])
        # need p_num points
        p_num = get_points_num(predicate)
        # get points of Entity, include old and new defined points
        new_points = self.distribute_entity_points(p_num)
        # replace ABC.. -> distributed points
        clause_base, pred_info = replace_points(predicate, pred_info, new_points)
        # get construct and text clauses
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        text_cdl = [clause_base]
        # add constraints: ab \\parallel cd ..
        self.polygons = append_lst(self.polygons, [tuple(new_points)])
        all_extend = self.get_all_extend(pred_info['extend'])
        if len(all_extend) == 0:
            all_extend = [clause_base]
        self.constraints += all_extend
        self.constraints_base += all_extend
        self.update_constraints_for_points(all_extend)
        return constr_cdl, text_cdl

    
    def define_relation(self, pred_name):
        '''try to distribute points for (Find), and construct new points in (Construct)'''
        constr_cdls = []
        text_cdls = []
        predicate = self.predicate_rel_names[pred_name]
        # IsMidpointOfLine(M,AB)
        pred_info = deepcopy(self.predicate_GDL["Relation"][predicate])
        # points mapping, from template to existed
        mapping = {}
        # try to distribute points for 'find'
        for clause in pred_info['find']:
            # distribute points according to clause
            ignore_ps = [v for k, v in mapping.items()]
            mapping_i, c_cdls, t_cdls = self.find_construct_clause(clause, pred_name, ignore_ps)
            # merge mapping
            # some points may be repeated
            for p, new_p in mapping_i.items():
                mapping[p] = new_p
                
            constr_cdls = append_lst(constr_cdls, c_cdls) 
            text_cdls = append_lst(text_cdls, t_cdls)
        # construct new points in 'construct'
        for point, fixed in zip(pred_info['construct'], pred_info['fixed']):
            if fixed:
                new_p = self.add_new_points(1)[0]
            else:
                new_p = self.find_unconstrained_points(
                    n=1, # ignore points that have existed in mapping.
                    ignore_ps=[v for k, v in mapping.items()]
                )[0]
            mapping[point] = new_p
        # replace template points -> distributed points
        clause_relation, pred_info = replace_points(
            predicate, pred_info, 
            new_points=None, mapping=mapping
        )
        # get construct and text cdls
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        constr_cdls += constr_cdl
        text_cdls += [clause_relation]
        # add constraints
        all_extend = self.get_all_extend(pred_info['extend'])
        # add collinear / cocircular / polygon (> 4) to all_extend (have algebra relation)
        for cdl in constr_cdls:
            if 'Collinear' in cdl or 'Cocircular' in cdl:
                all_extend.append(cdl)
            if 'Polygon' in cdl:
                _, items = parse_clause(cdl)
                if len(items[0]) > 3:
                    all_extend.append(cdl)
        if len(all_extend) == 0:
            all_extend = [clause_relation]
        self.constraints += all_extend
        self.update_constraints_for_points(all_extend)
        return constr_cdls, text_cdls
    
    def define_collinear_cocircular(self, pred_name):
        '''define collinear, cocircular'''
        construct_cdls = []
        text_clds = []
        if pred_name == 'Collinear':
            new_lines = self.distribute_collinear_points()
            for new_line in new_lines:
                cdl = f"Collinear({''.join(new_line)})"
                target = new_line[0]
                construct_cdls.append(cdl)
                self.update_constraints_for_points([cdl], [[target]])

        elif pred_name == 'Cocircular':
            circle, points_oncircle = self.distribute_cocircular_points()
            if len(points_oncircle) != 0:
                cdl = f"Cocircular({circle},{''.join(points_oncircle)})"
                construct_cdls.append(cdl)
                self.update_constraints_for_points([cdl])
                
        return construct_cdls, text_clds
    
    def find_construct_clause(self, clause, pred_name, ignore_ps=[],):
        '''find points for clause (template), try to return existed points'''
        constr_cdls = []
        text_cdls = []
        if 'Line' in clause:
            if len(self.lines) == 0:
                self.add_new_line()
            lines = [l for l in self.lines if len(l)==2]
            lines = [l for l in lines if not any([p in l for p in ignore_ps])]
            points = list(random.choice(self.lines))
            
        if 'Circle' in clause:
            if len(self.circles) == 0:
                self.add_new_circle()
            points = self.circles[:1]
        
        if 'Collinear' in clause:
            if all([len(l)<3 for l in self.lines]):
                c_cdls, _ = self.define_collinear_cocircular('Collinear')
                constr_cdls += c_cdls
            collinear_ls = [l for l in self.lines if len(l) ==3]
            points = list(random.choice(collinear_ls))
        
        if 'Cocircular' in clause:
            if len(self.circles) == 0:
                _, t_cdls = self.define_collinear_cocircular('Cocircular')
                text_cdls += t_cdls
                
            items = clause.lstrip('Cocircular(').rstrip(')')
            num = len(items.split(',')[-1])
            if len(self.circles) == 0:
                self.add_new_circle()
            circle = random.choice(self.circles)
            ori_ps_on_circle = self.points_on_circle[circle]
            # x<3, create points up to 3
            if len(ori_ps_on_circle) < 3:
                ps_to_sample = [p for p in self.points 
                                if p != circle and p not in ori_ps_on_circle
                                and p not in self.find_unconstrained_points()]
                if len(ps_to_sample) < 3 - len(ori_ps_on_circle):
                    ps_to_sample = ps_to_sample + self.add_new_points(
                        3-len(ori_ps_on_circle)-len(ps_to_sample)
                    )
                comb_list = list(itertools.combinations(ps_to_sample, 3-len(ori_ps_on_circle)))
                if len(comb_list) == 0:
                    new_ps_on_circle = self.add_new_points(
                            3-len(ori_ps_on_circle)
                        )
                else:
                    random.shuffle(comb_list)
                    for new_ps_on_circle in comb_list:
                        # remove collinear points
                        if not self.check_collinear(list(new_ps_on_circle) + ori_ps_on_circle):
                            break
                self.points_on_circle[circle] += list(new_ps_on_circle)
                
            # y>x, create y-x new points
            if num > len(self.points_on_circle[circle]):
                new_ps_on_circle = self.add_new_points(
                    num-len(self.points_on_circle[circle])
                )
                self.points_on_circle[circle] += new_ps_on_circle
                
            constr_cdls = append_lst(constr_cdls, [f"Cocircular({circle},{''.join(self.points_on_circle[circle])})"])
            points = random.sample(self.points_on_circle[circle], num)
            points = [circle] + points
        
        if 'Arc' in clause:
            if len(self.circles) == 0:
                _, t_cdls = self.define_collinear_cocircular('Cocircular')
                text_cdls += t_cdls
            items = clause.lstrip('Arc(').rstrip(')')
            num = len(items[1:])
            if len(self.circles) == 0:
                self.add_new_circle()
            # if not enough points on circle, create points
            # 1. x points on circle, x<3, find 3-x existed point and add
            # 2. x points on circle, x >=3, pass
            # 3. sample y points, y<=x, direct sample
            # 4. sample y points, y>x, create y-x new points then sample
            circle = random.choice(self.circles)
            ori_ps_on_circle = self.points_on_circle[circle]
            # x<3, create points up to 3
            if len(ori_ps_on_circle) < 3:
                ps_to_sample = [p for p in self.points 
                                if p != circle and p not in ori_ps_on_circle
                                and p not in self.find_unconstrained_points()]
                if len(ps_to_sample) < 3 - len(ori_ps_on_circle):
                    ps_to_sample = ps_to_sample + self.add_new_points(
                        3-len(ori_ps_on_circle)-len(ps_to_sample)
                    )
                comb_list = list(itertools.combinations(ps_to_sample, 3-len(ori_ps_on_circle)))
                if len(comb_list) == 0:
                    new_ps_on_circle = self.add_new_points(
                            3-len(ori_ps_on_circle)
                        )
                else:
                    random.shuffle(comb_list)
                    for new_ps_on_circle in comb_list:
                        # remove collinear points
                        if not self.check_collinear(list(new_ps_on_circle) + ori_ps_on_circle):
                            break
                self.points_on_circle[circle] += list(new_ps_on_circle)
                
            # y>x, create y-x new points
            if num > len(self.points_on_circle[circle]):
                new_ps_on_circle = self.add_new_points(
                    num-len(self.points_on_circle[circle])
                )
                self.points_on_circle[circle] += new_ps_on_circle
                
            constr_cdls += [f"Cocircular({circle},{''.join(self.points_on_circle[circle])})"]

            # can't choose arc that corresponds to diameter
            points_combs = [
                list(comb) for comb in 
                itertools.combinations(self.points_on_circle[circle], num)
            ]
            if self.predicates_base[0] in ['Rectangle', 'Square']:
                points_combs = [comb for comb in points_combs 
                                if comb not in [['a','c'], ['b','d']]]
            if self.predicates_base[0] == 'RightTrapezoid':
                points_combs = [comb for comb in points_combs 
                                if comb != ['b','d']]
            if self.predicates_base[0] == 'RightTriangle':                 
                points_combs = [comb for comb in points_combs 
                                if comb != ['a','c']]
            if self.predicates_base[0] == 'IsoscelesRightTriangle':
                points_combs = [comb for comb in points_combs 
                                if comb != ['b','c']]
            # points = random.sample(self.points_on_circle[circle], num)
            points = random.choice(points_combs)
            points = [circle] + points
         
        if 'Polygon' in clause:
            items = clause.lstrip('Polygon(').rstrip(')')
            if self.p_num < 3:
                self.add_new_points(3 - self.p_num)
            polygons = [p for p in self.polygons if len(p)==len(items)]
            if len(polygons) == 0:
                if len(items) == 3:
                    tris = list(itertools.combinations(self.points, 3))
                    collinear_ps = [l for l in self.lines if len(l) >= 3]
                    for collinear_p in collinear_ps:
                        if collinear_p in tris:
                            tris.remove(collinear_p)
                    ps = random.choice(tris)
                elif len(items) == 4:
                    tris = [p for p in self.polygons if len(p)==3]
                    if len(tris) == 0:
                        tris = list(itertools.combinations(self.points, 3))
                        collinear_ps = [l for l in self.lines if len(l) >= 3]
                        for collinear_p in collinear_ps:
                            if collinear_p in tris:
                                tris.remove(collinear_p)
                        ps = random.choice(tris)
                    ps = random.choice(tris)
                    new_p = self.add_new_points(1)[0]
                    ps = tuple(list(ps) + [new_p])
                self.define_construct_clauses([f"Polygon({''.join(ps)})"])
                constr_cdls += [f"Polygon({''.join(ps)})"]
                
            polygons = [p for p in self.polygons if len(p)==len(items) and p not in self.polygons_used]
            points = list(random.choice(polygons))
            self.polygons_used.append(points)
            if pred_name == 'IsAltitudeOfTriangle':
                # IsAltitudeOfTriangle(AD, ABC), angle ABC / ACB can't be 90
                # if == 90, move right angle point to the first 
                cdl_1 = f"Equal(MeasureOfAngle({''.join(points)}),90)"
                cdl_2 = f"Equal(MeasureOfAngle({''.join([points[2], points[0], points[1]])}),90)"
                if cdl_1 in self.constraints or cdl_2 in self.constraints: 
                    points = [points[1], points[2], points[0]]
   
                cdl_1 = f"Equal(MeasureOfAngle({''.join([points[0], points[2], points[1]])}),90)"
                cdl_1 = f"Equal(MeasureOfAngle({''.join([points[1], points[2], points[0]])}),90)"
                if cdl_1 in self.constraints or cdl_2 in self.constraints:
                    points = [points[2], points[0], points[1]]
        
        if 'Angle' in clause:
            angles = []
            for p in self.points:
                lines = [l for l in self.lines if p in l]
                if len(lines) >= 2:
                    lines = random.sample(lines, 2)
                    angles.append(lines)

            two_lines = random.choice(angles)
            p_mid = set(two_lines[0]).intersection(two_lines[1])
            p_mid = list(p_mid)[0]
            p1 = list(set(two_lines[0]) - set([p_mid]))[0]
            p2 = list(set(two_lines[1]) - set([p_mid]))[0]
            points = [p1, p_mid, p2]
        
        if 'Point' in clause:
            points = [p for p in self.points if p not in ignore_ps]
            if len(points) == 0:
                points = self.add_new_points(1)
            points = [random.choice(points)]
        
        # create mapping
        items = clause.split('(')[-1].rstrip(')')
        if ',' in items:
            items = items.replace(',', '')
        template_ps = list(items)
        mapping = {}
        for template_p, new_p in zip(template_ps, points):
            mapping[template_p] = new_p
            
        return mapping, constr_cdls, text_cdls
          
    def define_construct_clauses(self, clauses):
        '''define clauses directly'''
        constr_cdls = []
        for clause in clauses:
            if 'Polygon' in clause:
                points = get_content(clause)
                lines = [f"{points[i]}{points[(i+1)%len(points)]}" for i in range(len(points))]
                constr_cdls.append(f"Shape({','.join(lines)})")
                # constr_cdls.append(clause)
                for l in lines:
                    self.add_new_line(tuple(l))
                self.polygons = append_lst(self.polygons, [tuple(points)])
                
            elif 'Point' in clause:
                points = get_content(clause)
                self.points = append_lst(self.points, points)
                
            elif 'Line' in clause:
                points = get_content(clause)
                assert len(points) == 2
                constr_cdls.append(f"Shape({''.join(points)})")
                self.add_new_line(tuple(points))
                
            elif 'Angle' in clause:
                points = get_content(clause)
                line_1 = tuple([points[0], points[1]])
                line_2 = tuple([points[1], points[2]])
                self.add_new_line(line_1)
                self.add_new_line(line_2)
                
            elif 'Arc' in clause:
                points = get_content(clause)
                circle_p = clause.split('Arc(')[-1].rstrip(')')[0]
                others_p = clause.split('Arc(')[-1].rstrip(')')[1:]
                self.points = append_lst(self.points, points)
                if circle_p not in self.circles:
                    self.add_new_circle(circle_p)
                self.points_on_circle[circle_p] = append_lst(self.points_on_circle[circle_p], others_p)

                
            elif 'Collinear' in clause:
                points = get_content(clause)
                constr_cdls.append(clause)
                self.add_new_line(tuple(points))
                
            elif 'Cocircular' in clause:
                constr_cdls.append(clause)
                circle_p = clause.split('Cocircular(')[-1].rstrip(')').split(',')[0]
                others_p = list(clause.split('Cocircular(')[-1].rstrip(')').split(',')[1])
                if circle_p not in self.circles:
                    self.add_new_circle(circle_p)
                self.points_on_circle[circle_p] = append_lst(self.points_on_circle[circle_p], others_p)
                
            elif 'Circle' in clause:
                circles = get_content(clause)
                self.points = append_lst(self.points, circles)
                self.circles = append_lst(self.circles, circles)
            else:
                print('ERROR: ', clause)
                raise KeyError(clause)
                
        return constr_cdls
    
    def distribute_entity_points(self, n):
        '''找n个点'''
        if n > self.p_num:
            new_points = self.add_new_points(n - self.p_num)
            new_points = sorted(list(set(self.points + new_points)))
        else:
            new_points = sorted(random.sample(self.points, n))

        return new_points
        
    
    def distribute_collinear_points(self):
        '''find unconstrained / new point, define it on existed line'''
        if self.l_num == 0:
            if self.p_num < 2:
                self.add_new_points(2)
            points = random.sample(self.points, 2)
            self.add_new_line(tuple(points))
        
        # create: unconstrained point A, line BC -> line ABC
        new_lines = []
        unconstrained_ps = self.find_unconstrained_points()
        if len(unconstrained_ps) == 0:
            unconstrained_ps = self.find_unconstrained_points(n=1)
        for A in unconstrained_ps:
            # if there's one angle constraints, pass because too complex
            if len(self.points_with_constraints[A]) == 1:
                continue
            avai_lines = [sorted(l) for l in self.lines]
      
            # remove parallel lines
            ori_line = [l for l in self.lines if A in l]
            if len(ori_line) >= 1:
                ori_line = ori_line[0]
                parallel_lines = self.find_parallel(ori_line)
                avai_lines = [l for l in avai_lines if l not in parallel_lines]
            
            if len(avai_lines) > 0:
                BC = random.choice(avai_lines)
                new_line = tuple([A, BC[0], BC[1]])
                new_lines.append(new_line)
                self.add_new_line(new_line)
        
        return new_lines 
            
    def distribute_cocircular_points(self):
        '''find unconstrained / new point, define it on circle'''
        if len(self.circles) == 0:
            circle = self.add_new_circle()
            new_ps = random.sample(self.points, k=3)
            self.points_on_circle[circle] = new_ps 
            
        else:
            circle = self.circles[0]
            ps_on_c = self.points_on_circle[circle]
            new_ps = self.find_unconstrained_points(ignore_ps=ps_on_c)
             
            self.points_on_circle[circle] = append_lst(
                self.points_on_circle[circle], new_ps
            )

        return self.circles[0], new_ps

    def add_new_points(self, n):
        new_ps = []
        for i in range(self.p_num, n+self.p_num):
            new_ps.append(self.all_letters[i])
        
        self.points += new_ps
        
        for p in new_ps:
            self.points_with_constraints[p] = []
                
        return new_ps
    
    def add_new_line(self, item=None):
        # no item input
        if item == None:
            new_ps = self.add_new_points(2)
            item = tuple([new_ps[0], new_ps[1]])
        # single point
        if len(item) == 1:
            new_p = self.add_new_points(1)[0]
            item = tuple([item[0], new_p])
        if type(item) != tuple:
            item = tuple(item)
            
        item = tuple(sorted(item))
        # line A B C, remove line A B
        if len(item) > 2:
            for (i, j) in list(itertools.combinations(item, 2)):
                if tuple([i, j]) in self.lines:
                    self.lines.remove(tuple([i, j]))
                
        if item not in self.lines:
            if tuple([item[1], item[0]]) not in self.lines:
                self.lines.append(item)
        return item
    
    def add_new_circle(self, item=None):
        # no item input
        if item == None:
            item = self.add_new_points(1)[0]
        # single point
        if len(item) == 1:
            if type(item) == tuple:
                item = item[0]
            self.circles.append(item)
            self.points_on_circle[item] = []
        return item
    
    def check_collinear(self, points):
        if len(points) == 2:
            return False
        lines = itertools.permutations(points)
        for l in lines:
            if l in self.lines:
                return True
        return False
    
    def find_parallel(self, line):
        para_clauses = [c for c in self.constraints if 'ParallelBetweenLine' in c]
        parallels = []
        for clause in para_clauses:
            _, items = parse_clause(clause)
            line_1, line_2 = items
            if all([p in line_1 for p in line]):
                parallels.append(sorted(line_2))
            if all([p in line_2 for p in line]):
                parallels.append(sorted(line_1))
        
        perps = self.find_perpendicular(line)
        for perp_line in perps:
            perps_of_perp = self.find_perpendicular(perp_line)
            if sorted(line) in perps_of_perp:
                perps_of_perp.remove(sorted(line))
            if len(perps_of_perp) != 0:
                parallels += perps_of_perp
                
        return parallels
            
    def find_perpendicular(self, line):
        perp_clauses = [c for c in self.constraints if 'MeasureOfAngle' in c and '90' in c]

        perps = []
        for clause in perp_clauses:
            _, items = parse_clause(clause)
            angle = items[0]
            line_1, line_2 = [angle[0], angle[1]], [angle[1], angle[2]]
            if all([p in line_1 for p in line]):
                perps.append(sorted(line_2))
            if all([p in line_2 for p in line]):
                perps.append(sorted(line_1))
        
        return perps
    

def test():
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")

    for i in range(100):
        # predicates_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
        # predicates_rel = random.choices(PREDICATES_REL, k=2)
        
        predicates_base = ['MirrorSimilarBetweenTriangle']
        predicates_rel = ['IsMedianOfTriangle']
        
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            predicates_base, 
            predicates_rel, 
            n_new_lines=2
        )
        print('---------- Chosen Predicates ----------')
        print('predicates_base: ', predicates_base)
        print('predicates_rel: ', predicates_rel)
        
        
        print('---------- Construct CDLs ----------')
        for c_cdl in c_cdls:
            print(c_cdl)
        
        print('---------- Text CDLs ----------')
        for t_cdl in t_cdls:
            print(t_cdl)
        
        print('---------- States ----------')
        cg.print_states()
        print('=====================================')

def get_theorem_keys(t_info, theorem_GDL):
    theorem_keys = []
    for t in theorem_GDL:
        t_name = t.split('(')[0]
        # split
        if 'similar_arc' in t_name:
            continue
        if t_info[t_name][1] > 0:
            theorem_keys.append(t)
            
    return theorem_keys

def parse_premise_str(premise):
    clauses = premise.split('&')
    clauses_res = []
    for clause in clauses:
        if '|' in clause:
            clause = clause.split('|')[0]
            if clause.startswith('('):
                clause = clause[1:]
            if clause.endswith(')'):
                clause = clause[:-1]
        clauses_res.append(clause)
        
    return clauses_res


def process_equal_clause(clause):
    assert 'Equal' in clause
    pattern = r'(?<=[(,])[A-Z]+(?=[,)])'  # 匹配括号/逗号间的大写字母序列
    replaced = re.sub(
        pattern,
        lambda m: 'X' * len(m.group()),
        clause
    )
    return replaced

def get_clause_template(clause):
    pattern = r'(?<=[(,])[A-Z]+(?=[,)])'  # 匹配括号/逗号间的大写字母序列
    replaced = re.sub(
        pattern,
        lambda m: 'X' * len(m.group()),
        clause
    )
    return replaced

def get_para_from_clauses(clauses):
    pattern = r'(?<=[(,])[A-Z]+(?=[,)])'
    points = set()
    for clause in clauses:
        matches = re.findall(pattern, clause)
        for match in matches:
            points.update(set(list(match)))
    return list(points)

def replace_points_for_clause(clause, p_map):  
    pattern = r'(?<=[(,])[A-Z]+(?=[,)])'
    
    def repl(match):
        # match.group(0) 获取匹配到的变量（例如 "AC"）
        var = match.group(0)
        # 对变量中的每个字母进行替换
        return ''.join(p_map.get(letter, letter) for letter in var)
    
    # 使用 re.sub 进行替换
    return re.sub(pattern, repl, clause)

def parse_expression(expr):
    expr = expr.strip()
    if '(' in expr and expr.endswith(')'):
        name_part, rest = expr.split('(', 1)
        name = name_part.strip()
        args_str = rest[:-1]
        args = split_args(args_str)
        parsed_args = []
        for arg in args:
            parsed_arg = parse_expression(arg)
            parsed_args.append(parsed_arg)
        return {'name': name, 'args': parsed_args}
    else:
        return list(expr)

def split_args(args_str):
    args = []
    current = []
    stack = []
    for c in args_str:
        if c == '(':
            stack.append('(')
        elif c == ')':
            if stack:
                stack.pop()
        elif c == ',' and not stack:
            args.append(''.join(current).strip())
            current = []
            continue
        current.append(c)
    if current:
        args.append(''.join(current).strip())
    return args

def collect_mappings(source, target, mappings):
    if isinstance(source, dict) and isinstance(target, dict):
        if source['name'] != target['name']:
            return False
        if len(source['args']) != len(target['args']):
            return False
        for s_arg, t_arg in zip(source['args'], target['args']):
            if not collect_mappings(s_arg, t_arg, mappings):
                return False
        return True
    elif isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            return False
        for s_elem, t_elem in zip(source, target):
            if not collect_mappings(s_elem, t_elem, mappings):
                return False
        return True
    elif isinstance(source, str) and isinstance(target, str):
        if len(source) != 1 or len(target) != 1:
            return False
        t_char = target
        s_char = source
        if t_char in mappings:
            if mappings[t_char] != s_char:
                return False
        else:
            mappings[t_char] = s_char
        return True
    else:
        return False

def find_point_map(source_clause, target_clause):
    try:
        source_tree = parse_expression(source_clause)
        target_tree = parse_expression(target_clause)
    except:
        return None
    
    mappings = {}
    if not collect_mappings(source_tree, target_tree, mappings):
        return None
    
    values = list(mappings.values())
    if len(values) != len(set(values)):
        return None
    
    return mappings


def check_chain(conclusion_pre, premise_cur):
    # check: 
    # conclusion of the previous theorem is in premise of next theorem

    if type(conclusion_pre) in [list, tuple]:
        # 只取第一个结论
        conclusion_pre = conclusion_pre[0]
    # if 'Equal' in conclusion_previous:
    #     conclusion_name = process_equal_clause(conclusion_previous)
    # else:
    #     conclusion_name = conclusion_previous.split('(')[0]
    conclusion_pre_temp = get_clause_template(conclusion_pre)
    # conclusion_next_temp = get_clause_template(conclusion_next)
    # premise_pre_temp = [get_clause_template(c) for c in parse_premise_str(premise_pre)]
    premise_cur_temp = [get_clause_template(c) for c in parse_premise_str(premise_cur)]
    f1 = conclusion_pre_temp in premise_cur_temp
    p_map = None
    # f2 = conclusion_next_temp not in premise_pre_temp
    if f1:
        premises = parse_premise_str(premise_cur)
        target_premise = premises[premise_cur_temp.index(conclusion_pre_temp)]
        # try to build point mapping
        p_map = find_point_map(conclusion_pre, target_premise)
        if p_map is None:
            f1 = False
        
    return f1, p_map

# def build_theorem_chains_subprocess(t_chain, theorem_GDL, length):
    

def process_batch(batch, theorem_GDL, length, batch_size):
    results = []
    for t_chain in tqdm(batch, total=batch_size):
        t_premises = deepcopy([theorem_GDL[t]["1"]["premise"] for t in t_chain])
        t_conclusions = deepcopy([theorem_GDL[t]["1"]["conclusion"][0] for t in t_chain])
        is_chain = True
        t_chain_info = {
            t_chain[0]: {
                "premise": t_premises[0],
                "conclusion": t_conclusions[0]
            }
        }
        p_map = {}
        unused_ps = list(string.ascii_uppercase)
        conclsion_previous = t_conclusions[0]
        for i in range(1, length):
            # check chain: whether previous conclusion in current premise
            is_chain_i, p_map_i = check_chain(
                conclsion_previous, 
                t_premises[i], 
            )
            
            if is_chain_i:
                p_map.update(p_map_i)
                
                # find points to replace
                ps_current = get_para_from_clauses(
                    parse_premise_str(t_premises[i]))
                # ps_previous = get_para_from_clauses(
                    # parse_premise_str(t_premises[i-1]))
                
                # unused ps: not in previous step
                
                # update p_map: current step -> previous step
                for p in ps_current:
                    if p in p_map: 
                        continue
                    if p not in list(p_map.values()):
                        # p not seen in previous step
                        p_map[p] = p
                    else:
                        # p seen in previous step, use an unused p
                        unused_ps = [p for p in unused_ps if p not in list(p_map.values())]
                        p_map[p] = unused_ps.pop(0)
                            
                # replace points according to p_map
                premise_clauses_repl = []
                for clause in parse_premise_str(t_premises[i]):
                    clause_repl = replace_points_for_clause(clause, p_map)
                    premise_clauses_repl.append(clause_repl)
                    
                t_premise_repl = '&'.join(premise_clauses_repl)
                t_conclusion_repl = replace_points_for_clause(t_conclusions[i], p_map)

                t_chain_info[t_chain[i]] = {
                    "premise": t_premise_repl,
                    "conclusion": t_conclusion_repl
                }
            else:
                is_chain = False
                break
                
        if is_chain:
            results.append(t_chain_info)
    return results

def build_theorem_chains(theorem_GDL, t_info, length=2, debug=False):
    # 搜索定理链，实际上就是在构造解题过程
    # 如何处理相等的predicates？
    import math
    import os
    from multiprocessing import Pool
    from utils.tools import read_json, write_json
    theorem_keys = get_theorem_keys(t_info, theorem_GDL)
    # parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
    # parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, parsed_predicate_GDL)
    if length == 1:
        return theorem_keys
    else:
        file_name = f"theorem_chains_{length}.json"
        os.makedirs('json', exist_ok=True)
        if os.path.exists(f'json/{file_name}'):
            data = read_json(f'json/{file_name}')
            # return data
    
        
        t_chains = itertools.permutations(theorem_keys, length)
        iter_cnt = math.factorial(len(theorem_keys)) // math.factorial(len(theorem_keys) - length)
        
        num_process = 12
        pool = Pool(num_process)
        
        from functools import partial

        
        # 将t_chains分块，使用迭代器的分块而不是列表
        batch_size = iter_cnt // num_process
        batches = []
        for i in range(num_process):
            batch = itertools.islice(t_chains, i*batch_size, (i+1)*batch_size)
            # batch = list(batch)
            batches.append(batch)

        # 提交批次任务

        if debug:
            batch_results = []
            for batch in batches:
                batch_result = process_batch(batch, theorem_GDL, length, batch_size)
                batch_results.append(batch_result)
        else:
            with Pool(processes=num_process) as pool:
                batch_results = pool.map(
                    partial(
                        process_batch, 
                        theorem_GDL=theorem_GDL, 
                        length=length, 
                        batch_size=batch_size
                    ), 
                    batches
                )

        t_chains_select = []
        for batch_result in batch_results:
            t_chains_select += batch_result

        save_dict = {i: t_chain for i, t_chain in enumerate(t_chains_select)}
            
        write_json(f'json/{file_name}', save_dict)
        print('Before: ', iter_cnt)
        print('After : ', len(t_chains_select))
        return t_chains_select
        

        
def test_2():
    import json
    predicate_GDL = json.load(open('json/predicate_GDL_for_search.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open(
        # 'json/theorem_GDL_for_search.json', 
        'json/theorem_GDL.json', 
        'r', encoding='utf-8'))
    t_info = json.load(open(
        # 'json/t_info_new.json', 
        'json/t_info.json', 
        'r', encoding='utf-8'))
    cg = ClauseGenerator(predicate_GDL, theorem_GDL)
    
    theorem_chains = build_theorem_chains(
        theorem_GDL, t_info, 3,
        debug=False                          
    )
    print('total theorem chains num: ', len(theorem_chains))
    
    for theorem_chain in theorem_chains:
        print('---------- Chosen Theorems ----------')
        for t in theorem_chain:
            print(t)
    
        cg.generate_clauses_from_theorems(
            theorem_chain
        )
    
    
if __name__ == '__main__':
    setup_seed(1234)
    # test()
    test_2()