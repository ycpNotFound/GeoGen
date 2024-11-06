import time
import random
from formalgeo.problem import Problem
from formalgeo.core import GeometryPredicateLogicExecutor as GPLExecutor
from formalgeo.core import EquationKiller as EqKiller
from formalgeo.parse import parse_predicate_gdl, parse_theorem_gdl, parse_problem_cdl
from formalgeo.tools import get_used_pid_and_theorem, debug_print
from copy import deepcopy
from inter_gps_solver.extended_definition import ExtendedDefinition
from inter_gps_solver.logic_parser import LogicParser
from inter_gps_solver.logic_solver import LogicSolver
from utils import parse_clause, formalgeo_to_intergps

def get_p2t_map_fw(t_info, parsed_theorem_GDL):
    """
    Predicate-theorem mapping hash table for Search Accelerating.
    :param t_info: <dict>, {t_name: (category_id, usage_count)}, user customization.
    :param parsed_theorem_GDL: parsed theorem GDL.
    """
    p2t_map_fw = {}
    for t_name in t_info:
        if t_info[t_name][1] == 0 or t_info[t_name][0] == 3:  # skip no used and diff t
            continue
        for t_branch in parsed_theorem_GDL[t_name]["body"]:
            theorem_unit = parsed_theorem_GDL[t_name]["body"][t_branch]
            premises = list(theorem_unit["products"])
            premises += list(theorem_unit["logic_constraints"])
            premises += list(theorem_unit["attr_in_algebra_constraints"])
            for predicate, p_vars in premises:
                if predicate[0] == "~":  # skip oppose
                    continue
                if predicate not in p2t_map_fw:
                    p2t_map_fw[predicate] = [(t_name, t_branch, p_vars)]
                elif (t_name, t_branch, p_vars) not in p2t_map_fw[predicate]:
                    p2t_map_fw[predicate].append((t_name, t_branch, p_vars))
    return p2t_map_fw


class FormalGeoSolver:

    def __init__(self, predicate_GDL, theorem_GDL, strategy, max_depth, beam_size, t_info, debug=False):
        """
        Initialize Forward Searcher.
        :param predicate_GDL: predicate GDL.
        :param theorem_GDL: theorem GDL.
        :param strategy: <str>, "dfs", "bfs", "rs", "bs".
        :param max_depth: max search depth.
        :param beam_size: beam search size.
        :param t_info: <dict>, {t_name: (category_id, usage_count)}, user customization.
        :param debug: <bool>, set True when need print process information.
        """
        self.parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, self.parsed_predicate_GDL)
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.strategy = strategy
        self.debug = debug
        self.p2t_map = get_p2t_map_fw(t_info, self.parsed_theorem_GDL)

        self.problem = None
        self.stack = None
        self.last_step = None
        self.step_size = None
        self.node_count = None  # {depth: node_count}

        self.problem_p_paras = None  # Perimeter
        self.problem_a_paras = None  # Area

    def init_search(self, problem_CDL):
        """Initial problem by problem_CDL and build root Node."""
        EqKiller.use_cache = True  # use cache to speed up solving
        EqKiller.cache_eqs = {}
        EqKiller.cache_target = {}

        timing = time.time()  # timing

        self.problem = Problem()  # init problem
        self.problem.load_problem_by_fl(
            self.parsed_predicate_GDL, self.parsed_theorem_GDL, parse_problem_cdl(problem_CDL))
        EqKiller.solve_equations(self.problem)
        self.problem.step("init_problem", 0)

        self.stack = []
        self.last_step = 0
        self.step_size = 0
        self.node_count = {1: 1}

        self.problem_p_paras = set()  # Perimeter
        self.problem_a_paras = set()  # Area
        for sym in self.problem.condition.attr_of_sym:
            predicate, paras = self.problem.condition.attr_of_sym[sym]
            if predicate.startswith("Perimeter"):
                for para in paras:
                    self.problem_p_paras.add(para)
            elif predicate.startswith("Area"):
                for para in paras:
                    self.problem_a_paras.add(para)

        debug_print(self.debug, "(pid={}, strategy={}, timing={:.4f}s) Initialize and start forward search...".format(
            problem_CDL["problem_id"], self.strategy, time.time() - timing))

        timing = time.time()
        selections = self.get_theorem_selection()
        self.add_selections([], selections)
        debug_print(self.debug, "(timing={:.4f}s) Expand {} child node.".
                    format(time.time() - timing, len(selections)))

    def search(self):
        """
        Search problem and return search result.
        :return solved: <bool>, indicate whether problem solved or not.
        :return seqs: <list> of <str>, solved theorem sequences.
        """
        self.leveled_condition = {}
        cur_depth = 0
        while len(self.stack) > 0:
            beam_count = len(self.stack)
            ratio = 1
            if cur_depth > 1:
                ratio = 2
                    
            if len(self.stack) > ratio * self.beam_size:  # select branch with beam size
                stack = []
                
                for i in random.sample(range(len(self.stack)), ratio * self.beam_size):
                    stack.append(self.stack[i])
                self.stack = stack
                beam_count = self.beam_size

            for i in range(beam_count):
                pos, selection = self.stack.pop(0)
                self.step_size += 1
                debug_print(self.debug, "\n(pos={}, node_count={}) Current node.".format(pos, self.node_count))
                timing = time.time()
                
                # save leveled condition, new added
                last_step = len(self.problem.condition.items)
                solved = self.apply_and_check_goal(selection)
                if len(self.problem.condition.items) - last_step > 0:
                    new_condition = deepcopy(self.problem.condition.items[last_step:])
                    self.leveled_condition[pos] = {}
                    for i in range(len(new_condition)):
                        self.leveled_condition[pos][last_step+i] = new_condition[i]
                    
                debug_print(self.debug, "(solved={}, timing={:.4f}s) Apply selection and check goal.".format(
                    solved, time.time() - timing))
                if solved is None:  # not update, close search branch
                    continue
                if solved:  # solved, return result
                    _, seqs = get_used_pid_and_theorem(self.problem)
                    return True, seqs
                else:  # continue search
                    if len(pos) == self.max_depth:
                        continue
                    timing = time.time()
                    selections = self.get_theorem_selection()
                    self.add_selections(pos, selections)
                    debug_print(self.debug, "(timing={:.4f}s) Expand {} child node.".
                                format(time.time() - timing, len(selections)))
            cur_depth += 1
            
        return False, None

    def get_theorem_selection(self):
        """
        Return theorem selections according to <self.last_step>.
        :return selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        selections = []

        timing = time.time()
        related_pres = []  # new added predicates
        related_syms = []  # new added/updated equations
        for step in range(self.last_step, self.problem.condition.step_count):  # get related conditions
            for _id in self.problem.condition.ids_of_step[step]:
                if self.problem.condition.items[_id][0] == "Equation":
                    for sym in self.problem.condition.items[_id][1].free_symbols:
                        if sym in related_syms:
                            continue
                        related_syms.append(sym)
                else:
                    if self.problem.condition.items[_id][0] not in self.p2t_map:
                        continue
                    item = self.problem.condition.items[_id][1]
                    for t_name, t_branch, p_vars in self.p2t_map[self.problem.condition.items[_id][0]]:
                        if len(p_vars) != len(item):
                            continue
                        letters = {}
                        for i in range(len(p_vars)):
                            letters[p_vars[i]] = item[i]
                        related_pre = (t_name, t_branch, letters)
                        if related_pre not in related_pres:
                            related_pres.append(related_pre)
        debug_print(self.debug, "(timing={:.4f}s) Get Related.".format(time.time() - timing))
        debug_print(self.debug, "Related predicates: {}.".format(related_pres))
        debug_print(self.debug, "Related syms: {}.".format(related_syms))

        timing = time.time()
        logic_selections = self.try_theorem_logic(related_pres)
        debug_print(self.debug, "(timing={:.4f}s) Get {} logic-related selections: {}.".format(
            time.time() - timing, len(logic_selections), logic_selections))
        timing = time.time()
        algebra_selections = self.try_theorem_algebra(related_syms)
        # algebra_selections = []
        debug_print(self.debug, "(timing={:.4f}s) Get {} algebra-related selections: {}.".format(
            time.time() - timing, len(algebra_selections), algebra_selections))

        timing = time.time()
        added_selections = []
        for selection in logic_selections + algebra_selections:  # remove redundancy
            _, conclusions = selection
            s = []
            for conclusion in conclusions:
                predicate, item, _ = conclusion
                s.append((predicate, item))
            s = tuple(s)
            if s not in added_selections:
                added_selections.append(s)
                selections.append(selection)

        for i in range(len(selections))[::-1]:
            t_msg, conclusions = selections[i]
            t_name, t_branch, t_para = t_msg
            if "area" in t_name:
                if "ratio" in t_name:
                    para1 = t_para[0:int(len(t_para) / 2)]
                    para2 = t_para[int(len(t_para) / 2):]
                    if not (para1 in self.problem_a_paras and para2 in self.problem_a_paras):
                        selections.pop(i)
                else:
                    if t_para not in self.problem_a_paras:
                        selections.pop(i)
            elif "perimeter" in t_name:
                if "ratio" in t_name:
                    para1 = t_para[0:int(len(t_para) / 2)]
                    para2 = t_para[int(len(t_para) / 2):]
                    if not (para1 in self.problem_p_paras and para2 in self.problem_p_paras):
                        selections.pop(i)
                else:
                    if t_para not in self.problem_p_paras:
                        selections.pop(i)
                        
        # sele_to_remove = []
        # for sele in selections:
        #     if sele[0][0] == 'round_angle':
        #         sele_to_remove.append(sele)
        # for sele in sele_to_remove:
        #     selections.remove(sele)
            
        debug_print(self.debug, "(timing={:.4f}s) Get {}  selections: {}.".format(
            time.time() - timing, len(selections), selections))

        return selections

    def try_theorem_logic(self, related_pres):
        """
        Try a theorem and return can-added conclusions.
        :param related_pres: <list>, list of tuple('t_name', 't_branch', letters).
        :return selections: <list> of ((t_name, t_branch, t_para, t_timing), ((predicate, item, premise))).
        """

        selections = []
        for t_name, t_branch, t_letters in related_pres:
            gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]
            results = GPLExecutor.run(gpl, self.problem, t_letters)  # get gpl reasoned result
            for letters, premise, conclusion in results:
                t_para = tuple([letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]])

                premise = tuple(premise)
                conclusions = []
                for predicate, item in conclusion:  # add conclusion
                    if self.problem.check(predicate, item, premise, t_name):
                        if predicate != "Equation":
                            item = tuple(item)
                        conclusions.append((predicate, item, premise))

                if len(conclusions) > 0:
                    selections.append(((t_name, t_branch, t_para), tuple(conclusions)))
                    if len(selections) > self.beam_size:
                        return selections
        # sele_to_remove = []
        # for sele in selections:
        #     if sele[0][0] == 'round_angle':
        #         sele_to_remove.append(sele)
        # for sele in sele_to_remove:
        #     selections.remove(sele)
            
        return selections

    def try_theorem_algebra(self, related_syms):
        """
        Try a theorem and return can-added conclusions.
        :param related_syms: <list>, related syms.
        :return selections: <list> of ((t_name, t_branch, t_para, t_timing), ((predicate, item, premise))).
        """
        paras_of_attrs = {}  # <dict>, {attr: [para]}
        for sym in related_syms:
            attr, paras = self.problem.condition.attr_of_sym[sym]
            if attr not in self.p2t_map:
                continue

            if attr not in paras_of_attrs:
                paras_of_attrs[attr] = []

            for para in paras:
                if para in paras_of_attrs[attr]:
                    continue
                paras_of_attrs[attr].append(para)

        selections = []
        for related_attr in paras_of_attrs:
            related_paras = set(paras_of_attrs[related_attr])
            for t_name, t_branch, p_vars in self.p2t_map[related_attr]:
                gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]  # run gdl
                for related_para in related_paras:
                    letters = {}
                    for i in range(len(p_vars)):
                        letters[p_vars[i]] = related_para[i]
                    results = GPLExecutor.run(gpl, self.problem, letters)  # get gpl reasoned result
                    for letters, premise, conclusion in results:
                        theorem_para = tuple([letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]])
                        premise = tuple(premise)
                        conclusions = []
                        for predicate, item in conclusion:  # add conclusion
                            if self.problem.check(predicate, item, premise, t_name):
                                if predicate != "Equation":
                                    item = tuple(item)
                                conclusions.append((predicate, item, premise))

                        if len(conclusions) > 0:
                            selections.append(((t_name, t_branch, theorem_para), tuple(conclusions)))
                            if len(selections) > self.beam_size:
                                return selections
        return selections

    def apply_and_check_goal(self, selection):
        """
        Apply selection and check goal.
        :param selection: ((t_name, t_branch, t_para), ((predicate, item, premise))).
        :return solved: <bool> or None. Set None when not update
        """
        self.last_step = self.problem.condition.step_count
        update = False
        t_msg, conclusions = selection

        for predicate, item, premise in conclusions:
            update = self.problem.add(predicate, item, premise, t_msg, skip_check=True) or update

        if not update:  # close current branch if applied theorem no new condition
            return None

        EqKiller.solve_equations(self.problem)  # solve eq & check_goal
        self.problem.check_goal()
        self.problem.step(t_msg, 0)

        return self.problem.goal.solved

    def add_selections(self, father_pos, selections):
        """
        Add selections to self.stack.
        :param father_pos: position of father branch.
        :param selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        pos = list(father_pos)
        depth = len(father_pos) + 1
        if depth not in self.node_count:
            self.node_count[depth] = 1

        for selection in selections:
            self.stack.append((tuple(pos + [self.node_count[depth]]), selection))
            self.node_count[depth] += 1



class InterGPSSolver:
    def __init__(self,
                 debug=True):
        self.debug = debug
        self.logic = None
        
    def prepare_for_intergps(self, problem_CDL):
        text_logic_forms = []
        for c in problem_CDL['text_cdl']:
            text_logic_forms += formalgeo_to_intergps(c)
           
        diagram_logic_forms = []
        for c in problem_CDL['construction_cdl'] + problem_CDL['image_cdl']:
            diagram_logic_forms += formalgeo_to_intergps(c)

        text_parser = {
            "text_logic_forms": text_logic_forms
        }
        lines = []
        for l in problem_CDL['line_instances']:
            if len(l) > 2:
                l = (l[0], l[-1])
            lines.append(''.join(l))
        diagram_parser = {
            "point_positions": problem_CDL['point_positions'],
            "line_instances": lines,
            "circle_instances": problem_CDL['circle_instances'],
            "diagram_logic_forms": diagram_logic_forms
        }
        return text_parser, diagram_parser
        
    
    def init_search(self, problem_CDL):
        text_parser, diagram_parser = self.prepare_for_intergps(problem_CDL)
        parser = LogicParser(ExtendedDefinition(debug=self.debug))
        parser.logic.point_positions = diagram_parser['point_positions']
        isLetter = lambda ch: ch.upper() and len(ch) == 1
        parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])
        if self.debug:
            print(parser.logic.point_positions)
        
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
                if self.debug:
                    print("The diagram logic form is", logic_form)
                try:
                    parse_tree = parser.parse(logic_form) # ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
                    parser.dfsParseTree(parse_tree)
                except Exception as e:
                    if self.debug:
                        print("\033[0;0;41mError:\033[0m", repr(e))
                        
        ## Parse text logic forms
        target = None
        text_logic_forms = text_parser["text_logic_forms"]
        for text in text_logic_forms:
            if self.debug:
                print("The text logic form is", text)
            if text.find('Find') != -1:
                target = parser.findTarget(parser.parse(text)) # ['Value', 'A', 'C']
            else:
                res = parser.parse(text)
                parser.dfsParseTree(res)

        self.logic = parser.logic

    def search(self):
        ## Set up, initialize and run the logic solver
        if self.logic is None:
            raise ValueError(self.logic)
        solver = LogicSolver(self.logic)
        solver.initSearch()
        answer, steps, step_lst = solver.beamSearch()

        return answer, steps, step_lst