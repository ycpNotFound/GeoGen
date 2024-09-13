import random
import re
import string
import itertools
from copy import deepcopy
from formalgeo.data import DatasetLoader
from utils import (PREDICATES_ATTR, PREDICATES_ENT, PREDICATES_PRE,
                   PREDICATES_REL, append_lst, get_content, get_points_num, get_symbol, get_predicate_name, get_points_mapping,
                   replace_points, setup_seed)



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
        self.constraints = []
        
        self.points_on_circle = {}

    
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
            "constraints": self.constraints
        }
    
    def empty_states(self):
        self.points = []
        self.lines = []
        self.circles = []
        self.constraints = []
        self.points_on_circle = {}
    
    def print_states(self):
        p_str = ', '.join(self.points)
        l_str = ', '.join([''.join(l)  for l in self.lines])
        c_str = ', '.join(self.circles)
        
        print(f"Points: {p_str}")
        print(f"Lines: {l_str}")
        print(f"Circles: {c_str}")
        print('Constraints: ')
        for c in self.constraints:
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
            constraints += self.get_extend(clause)
            
        return constraints
            
            
    def generate_clauses_from_predicates(self, n_entity, n_construct, n_relation):
        # define base entity (Square, Rectangle) 
        clauses_entity = random.choices(PREDICATES_ENT, k=n_entity)
        constr_cdls = [] # Shape, Collinear, Cocircular
        text_cdls = [] # Equal, Rectangle, IsMedianOf
        for entity_name in clauses_entity:
            constr_cdl, text_cdl = self.define_entity(entity_name)
            constr_cdls += constr_cdl
            text_cdls += text_cdl
        
        # define construct predicates (Collinear, Cocircular)
        clauses_constr = random.choices(PREDICATES_PRE, k=n_construct)
        for constr_name in clauses_constr:
            constr_cdl, text_cdl = self.define_construction(constr_name)
            constr_cdls += constr_cdl
            text_cdls += text_cdl
            
        # define relation predicates (Midpoint, Parallel ..)
        clauses_rel = random.choices(PREDICATES_REL, k=n_relation)
        for rel_name in clauses_rel:
            constr_cdl, text_cdl = self.define_relation(rel_name)
            constr_cdls += constr_cdl
            text_cdls += text_cdl
            
        return constr_cdls, text_cdls
            
    
    def define_entity(self, pred_name):
        '''define base entity (Square, Rectangle)'''
        pred_type = "Entity"
        predicate = self.predicate_ent_names[pred_name]
        # Predicate(ABC..)
        pred_info = deepcopy(self.predicate_GDL[pred_type][predicate])
        # need p_num points
        p_num = get_points_num(predicate)
        # get points of Entity, include old and new defined points
        new_points = self.distribute_new_points(p_num)
        # replace ABC.. -> distributed points
        predicate, pred_info = replace_points(predicate, pred_info, new_points)
        # get construct and text clauses
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        text_cdl = [predicate]
        # add constraints: ab \\parallel cd ..
        self.polygons = append_lst(self.polygons, [tuple(new_points)])
        all_extend = self.get_all_extend(pred_info['extend'])
        self.constraints += all_extend
        return constr_cdl, text_cdl
    
    def define_construction(self, pred_name):
        '''define collinear, cocircular'''
        construct_cdls = []
        text_clds = []
        if pred_name == 'Collinear':
            # extend 2 lines and make them intersect
            lines = self.distribute_new_lines()
            if len(lines[0]) > 2:
                cdl = f"Collinear({''.join(lines[0])})"
                construct_cdls.append(cdl)
            if len(lines[1]) > 2:
                cdl = f"Collinear({''.join(lines[1])})"
                construct_cdls.append(cdl)

        elif pred_name == 'Cocircular':
            # random choose 2 ~ 3 points, make circumcircle
            # create 1 new points on circle
            circle, points_oncircle, is_diameter = self.distribute_new_circle()
            c_cdl = f"Cocircular({circle},{''.join(points_oncircle)})"
            
            construct_cdls.append(c_cdl)
            
            if is_diameter:
                r_cdl = f"IsDiameterOfCircle({''.join(points_oncircle[:2])},{circle})"
                construct_cdls.append(f"Collinear({points_oncircle[0]}{circle}{points_oncircle[1]})")
                text_clds.append(r_cdl)
                constraint = f"IsMidpointOfLine({circle},{''.join(points_oncircle[:2])})"
                self.constraints += self.get_all_extend([constraint])
                
                
        return construct_cdls, text_clds
    
    def define_relation(self, pred_name):
        '''try to distribute points for (Find), and construct new points in (Construct)'''
        constr_cdls = []
        text_cdls = []
        pred_type = "Relation"
        predicate = self.predicate_rel_names[pred_name]
        # IsMidpointOfLine(M,AB)
        pred_info = deepcopy(self.predicate_GDL[pred_type][predicate])
        # points mapping, from template to existed
        mapping = {}
        # try to distribute points for 'find'
        for clause in pred_info['find']:
            # distribute points according to clause
            mapping_i, c_cdls, t_cdls = self.find_construct_clause(clause)
            # merge mapping
            # some points may be repeated
            for p, new_p in mapping_i.items():
                mapping[p] = new_p
                
            constr_cdls += c_cdls
            text_cdls += t_cdls
        # construct new points in 'construct'
        for point in pred_info['construct']:
            new_p = self.add_new_points(1)[0]
            mapping[point] = new_p
        # replace template points -> distributed points
        predicate, pred_info = replace_points(
            predicate, pred_info, 
            new_points=None, mapping=mapping
        )
        # get construct and text cdls
        constr_cdl = self.define_construct_clauses(pred_info['ee_check'])
        constr_cdls += constr_cdl
        text_cdls += [predicate]
        # add constraints
        all_extend = self.get_all_extend(pred_info['extend'])
        if len(all_extend) == 0:
            all_extend = [predicate]
        self.constraints += all_extend
        return constr_cdls, text_cdls
    
    def find_construct_clause(self, clause):
        # 找到clause对应点，尽可能返回已有的点，不构造新点
        constr_cdls = []
        text_cdls = []
        if 'Line' in clause:
            if len(self.lines) == 0:
                self.add_new_line()
            lines = [l for l in self.lines if len(l)==2]
            points = list(random.choice(self.lines))
            
        if 'Circle' in clause:
            if len(self.circles) == 0:
                self.add_new_circle()
            points = self.circles[:1]
        
        if 'Collinear' in clause:
            if all([len(l)<3 for l in self.lines]):
                c_cdls, _ = self.define_construction('Collinear')
                constr_cdls += c_cdls
            collinear_ls = [l for l in self.lines if len(l) ==3]
            points = list(random.choice(collinear_ls))
        
        if 'Cocircular' in clause:
            if len(self.circles) == 0:
                _, t_cdls = self.define_construction('Cocircular')
                text_cdls += t_cdls
                
            items = clause.lstrip('Cocircular(').rstrip(')')
            num = len(items.split(',')[-1])
            if len(self.circles) == 0:
                self.add_new_circle()
            # if there's no enough points on circle
            if len(self.points_on_circle[self.circles[0]]) < num:
                if num <= 3: # find 3 points to be on circle
                    ps_to_sample = [p for p in self.points if p != self.circles[0]]
                    ps_on_circle = random.sample(ps_to_sample, 3)
                    
                else: # if num > 3, create the rest points
                    ps_to_sample = [p for p in self.points if p != self.circles[0]]
                    ps_on_circle = random.sample(ps_to_sample, 3) + self.add_new_points(num - 3)
                
                self.points_on_circle[self.circles[0]] = append_lst(self.points_on_circle[self.circles[0]], ps_on_circle)
                
                constr_cdls += [f"Cocircular({self.circles[0]},{''.join(ps_on_circle)})"]
            points = random.sample(self.points_on_circle[self.circles[0]], num)
            points = [self.circles[0]] + points
        
        if 'Arc' in clause:
            if len(self.circles) == 0:
                _, t_cdls = self.define_construction('Cocircular')
                text_cdls += t_cdls
            items = clause.lstrip('Arc(').rstrip(')')
            num = len(items[1:])
            if len(self.circles) == 0:
                self.add_new_circle()
            if len(self.points_on_circle[self.circles[0]]) < num:
                if num <= 3: # find 3 points to be on circle
                    ps_to_sample = [p for p in self.points if p != self.circles[0]]
                    ps_on_circle = random.sample(ps_to_sample, 3)
                else: # if num > 3, create the rest points
                    ps_to_sample = [p for p in self.points if p != self.circles[0]]
                    ps_on_circle = random.sample(ps_to_sample, 3) + self.add_new_points(num - 3)
                self.points_on_circle[self.circles[0]] = append_lst(self.points_on_circle[self.circles[0]], ps_on_circle)
            points = random.sample(self.points_on_circle[self.circles[0]], num)
            points = [self.circles[0]] + points
         
        if 'Polygon' in clause:
            items = clause.lstrip('Polygon(').rstrip(')')
            if self.p_num < len(items):
                self.add_new_points(len(items) - self.p_num)
            polygons = [p for p in self.polygons if len(p)==len(items)]
            if len(polygons) == 0:
                ps = sorted(random.sample(self.points, len(items)))
                self.define_construct_clauses([f"Polygon({''.join(ps)})"])
            polygons = [p for p in self.polygons if len(p)==len(items)]
            points = list(random.choice(polygons))
        
        if 'Angle' in clause:
            angles = []
            for p in self.points:
                lines = [l for l in self.lines if p in l]
                if len(lines) > 2:
                    lines = random.sample(lines, 2)
                    angles.append(lines)

            two_lines = random.choice(angles)
            p_mid = set(two_lines[0]).intersection(two_lines[1])
            p_mid = list(p_mid)[0]
            p1 = list(set(two_lines[0]) - set([p_mid]))[0]
            p2 = list(set(two_lines[1]) - set([p_mid]))[0]
            points = [p1, p_mid, p2]
        
        if 'Point' in clause:
            points = [random.choice(self.points)]
        
        if 'Trapezoid' in clause:
            items = clause.lstrip('Trapezoid(').rstrip(')')
            mapping_i, _, _ = self.find_construct_clause(f"Polygon({items})")
            new_items = []
            for i in range(len(items)):
                new_items.append(mapping_i[items[i]])
            points = ''.join(new_items)
            text_cdls += [f"Trapezoid({''.join(points)})"]
        
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
        # 直接定义新的实体
        constr_cdls = []
        for clause in clauses:
            if 'Polygon' in clause:
                points = get_content(clause)
                lines = [f"{points[i]}{points[(i+1)%len(points)]}" for i in range(len(points))]
                constr_cdls.append(f"Shape({','.join(lines)})")
                self.lines = append_lst(self.lines, [tuple(line) for line in lines])
                self.polygons = append_lst(self.polygons, [tuple(points)])
                
            elif 'Point' in clause:
                points = get_content(clause)
                self.points = append_lst(self.points, points)
                
            elif 'Line' in clause:
                points = get_content(clause)
                assert len(points) == 2
                constr_cdls.append(f"Shape({''.join(points)})")
                self.lines = append_lst(self.lines, [tuple(points)])
                
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
                if tuple([points[0], points[-1]]) in self.lines:
                    self.lines.remove(tuple([points[0], points[-1]]))
                self.lines = append_lst(self.lines, [tuple(points)])
                
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
                
        return constr_cdls
    
    def distribute_new_points(self, n):
        '''找n个点'''
        if n > self.p_num + 1:
            new_points = self.add_new_points(n - self.p_num)
            new_points = sorted(list(set(self.points + new_points)))
        else:
            added_num = random.choice([1, 2])
            select_num = n - added_num
            selected_points = sorted(random.sample(self.points, select_num))
            new_points = selected_points + self.add_new_points(added_num)

        return new_points
        
    
    def distribute_new_lines(self):
        '''找2条未相交不平行的line，其中包含已有的、新创建的，使其相交'''
        if self.l_num == 0:
            if self.p_num < 2:
                self.add_new_points(2)
            points = random.sample(self.points, 2)
            self.add_new_line(tuple(points))
        
        # random choose one origin line
        line_chosen = random.choice(self.lines)
        lines_remain = self.find_lines_not_intersect_and_parallel(line_chosen, self.lines)
        points_remain = [p for p in self.points if p not in line_chosen]
        create_flag = random.choice([0,1])
        if len(lines_remain) == 0:
            create_flag = 1
        
        # create new line, point A, line BC -> line AD, line BDC
        if create_flag:
            ori_p = random.choice(points_remain)
            new_line = self.add_new_line(ori_p)
            new_p = new_line[-1]
            ori_line = tuple(list(line_chosen) + [new_p])
            new_lines = [ori_line, new_line]
        
        # extend exist lines, line AB, line CD -> line ABE, line CDE
        else:
            new_line = random.choice(lines_remain)
            new_p = self.add_new_points(1)[0]
            line1 = tuple(list(new_line) + [new_p])
            line2 = tuple(list(line_chosen) + [new_p])
            new_lines = [line1, line2]

        self.add_new_line(new_lines[0])
        self.add_new_line(new_lines[1])
        return new_lines
            
    def distribute_new_circle(self):
        '''以旧2点为直径创建三点共圆，或以旧3点作外接圆创建四点共圆'''
        # if there's already a circle, create up to 3 points or 1 new points
        if len(self.circles) != 0:
            if len(self.points_on_circle[self.circles[0]]) < 3:
                add_num = 3 - self.p_num
            else:
                add_num = 1
            new_points = self.add_new_points(add_num)
            circle = self.circles[0]
            self.points_on_circle[circle] = append_lst(self.points_on_circle[circle], new_points)
            return self.circles[0], new_points, False
        else: # create circle and points on circle
            circle = self.add_new_circle()
            if self.p_num < 3:
                new_points = self.add_new_points(3 - self.p_num)
                
            ori_points_num = random.choice([2, 3])
            if ori_points_num == 2: # sample 2 points as diameter
                chosen_points = random.sample(self.points, 2)
                new_points = chosen_points + self.add_new_points(1)
                is_diameter = True
            else: # sample 3 points
                while(1): # check if collinear
                    chosen_points = random.sample(self.points, 3)
                    collinear = False
                    for l in self.lines:
                        if all([p in l for p in chosen_points]):
                            collinear = True
                    if not collinear:
                        break
                new_points = chosen_points + self.add_new_points(1)
                is_diameter = False
            
            
            self.points_on_circle[circle] = append_lst(self.points_on_circle[circle], new_points) 
            return circle, new_points, is_diameter
             
    
    def add_new_points(self, n):
        new_ps = []
        for i in range(self.p_num, n+self.p_num):
            new_ps.append(self.all_letters[i])
        
        self.points += new_ps
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
        # line A B C, remove line A B
        if len(item) > 2:
            for (i, j) in list(itertools.combinations(item, 2)):
                if tuple([i, j]) in self.lines:
                    self.lines.remove(tuple([i, j]))
                
        if item not in self.lines:
            self.lines.append(item)
        return item
    
    def add_new_circle(self, item=None):
        # no item input
        if len(self.circles) == 1:
            return self.circles[0]
        if item == None:
            item = self.add_new_points(1)[0]
        # single point
        if len(item) == 1:
            if type(item) == tuple:
                item = item[0]
            self.circles.append(item)
            self.points_on_circle[item] = []
        return item
    
    def find_lines_not_intersect_and_parallel(self, target_line, lines):
        res_lines = []
        for l in lines:
            # do not have same letter
            if not bool(set(target_line) & set(l)): 
                # do not parallel
                para_flag = False
                for item in self.constraints:
                    if 'Parallel' in item and ''.join(l) in item and ''.join(target_line) in item:
                        para_flag = True
                if not para_flag:
                    res_lines.append(l)
                    
        return res_lines      
    
    

def test():
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    constraints = []
    for i in range(100):
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(1, 2, 2)
        
        print('---------- Construct CDLs ----------')
        for c_cdl in c_cdls:
            print(c_cdl)
        
        print('---------- Text CDLs ----------')
        for t_cdl in t_cdls:
            print(t_cdl)
        
        print('---------- States ----------')
        cg.print_states()
        print('=====================================')
        
        constraints += [get_predicate_name(s) for s in cg.constraints]
        constraints = list(set(constraints))
        
    print(constraints)
        
if __name__ == '__main__':
    setup_seed(1234)
    test()