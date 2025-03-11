import time
import json
import warnings

from formalgeo_v2.core import EquationKiller as EqKiller
from formalgeo_v2.core import GeometryPredicateLogicExecutor as GPLExecutor
from formalgeo_v2.parse import (get_equation_from_tree, parse_predicate_gdl,
                             parse_problem_cdl, parse_theorem_gdl)
from formalgeo_v2.problem import Problem
from formalgeo_v2.tools import debug_print, rough_equal
from formalgeo_v2.solver.forward_search import get_p2t_map_fw

class Interactor:

    def __init__(self, predicate_GDL, theorem_GDL, t_info=None, debug=False):
        """
        Initialize Interactor.
        :param predicate_GDL: predicate GDL.
        :param theorem_GDL: theorem GDL.
        """
        self.parsed_predicate_GDL = parse_predicate_gdl(predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(theorem_GDL, self.parsed_predicate_GDL)
        self.problem = None
        
        self.debug = debug
        self.last_step = 0
        if not t_info:
            t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
        self.p2t_map = get_p2t_map_fw(t_info, self.parsed_theorem_GDL)
        
        self.problem_p_paras = set()  # Perimeter
        self.problem_a_paras = set()  # Area
        self.ignore_ops = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle', 'Arc']

    def load_problem(self, problem_CDL):
        """Load problem through problem_CDL."""
        start_time = time.time()
        self.problem = Problem()
        self.problem.load_problem_by_fl(self.parsed_predicate_GDL,
                                        self.parsed_theorem_GDL,
                                        parse_problem_cdl(problem_CDL))  # load problem
        EqKiller.solve_equations(self.problem)  # Solve the equations after initialization
        self.problem.step("init_problem", time.time() - start_time)  # save applied theorem and update step

    def apply_theorem(self, t_name, t_branch=None, t_para=None):
        """
        Apply a theorem and return whether it is successfully applied.
        :param t_name: <str>.
        :param t_branch: <str>.
        :param t_para: tuple of <str>.
        :return update: <bool>, Whether the question condition updated or not.
        """
        # self.last_step = self.problem.condition.step_count
        
        if self.problem is None:
            e_msg = "Problem not loaded. Please run <load_problem> before run <apply_theorem>."
            raise Exception(e_msg)
        if t_name not in self.parsed_theorem_GDL:
            e_msg = "Theorem {} not defined in current GDL.".format(t_name)
            raise Exception(e_msg)
        if t_name.endswith("definition"):
            e_msg = "Theorem {} only used for backward reason.".format(t_name)
            raise Exception(e_msg)
        if t_para is not None and len(t_para) != len(self.parsed_theorem_GDL[t_name]["vars"]):
            e_msg = "Theorem <{}> para length error. Expected {} but got {}.".format(
                t_name, len(self.parsed_theorem_GDL[t_name]["vars"]), t_para)
            raise Exception(e_msg)
        if t_branch is not None and t_branch not in self.parsed_theorem_GDL[t_name]["body"]:
            e_msg = "Theorem <{}> branch error. Expected {} but got {}.".format(
                t_name, self.parsed_theorem_GDL[t_name]["body"].keys(), t_branch)
            raise Exception(e_msg)

        if t_para is None and t_branch is None:
            update = self.apply_theorem_by_name(t_name)
        elif t_para is not None and t_branch is None:
            update = self.apply_theorem_by_name_and_para(t_name, t_para)
        elif t_para is None and t_branch is not None:
            update = self.apply_theorem_by_name_and_branch(t_name, t_branch)
        else:
            update = self.apply_theorem_by_name_and_para_and_branch(t_name, t_branch, t_para)

        if not update:
            w_msg = "Theorem <{},{},{}> not applied. Please check your theorem or prerequisite.".format(
                t_name, t_branch, t_para)
            # warnings.warn(w_msg)

        return update

    def apply_theorem_by_name(self, t_name):
        """
        Apply a theorem with t_name.
        :param t_name: <str>.
        :return update: <bool>, Whether the problem condition updated or not.
        """
        update = False

        for branch in self.parsed_theorem_GDL[t_name]["body"]:
            timing = time.time()  # timing
            gpl = self.parsed_theorem_GDL[t_name]["body"][branch]

            conclusions = GPLExecutor.run(gpl, self.problem)  # get gpl reasoned result
            if len(conclusions) == 0:
                theorem = (t_name, branch, None)
                self.problem.step(theorem, time.time() - timing)
                continue
            avg_timing = (time.time() - timing) / len(conclusions)
            for letters, premise, conclusion in conclusions:
                t_para = [letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]]
                theorem = (t_name, branch, tuple(t_para))
                for predicate, item in conclusion:  # add conclusion
                    update = self.problem.add(predicate, item, premise, theorem) or update
                self.problem.step(theorem, avg_timing)

        timing = time.time()  # timing
        EqKiller.solve_equations(self.problem)
        self.problem.step("solve_equations", time.time() - timing)

        return update

    def apply_theorem_by_name_and_para(self, t_name, t_para):
        """
        Apply a theorem with t_name and t_para.
        :param t_name: <str>.
        :param t_para: tuple of <str>.
        :return update: <bool>, Whether the problem condition updated or not.
        """
        update = False
        letters = {}  # used for vars-letters replacement
        for i in range(len(self.parsed_theorem_GDL[t_name]["vars"])):
            letters[self.parsed_theorem_GDL[t_name]["vars"][i]] = t_para[i]

        for branch in self.parsed_theorem_GDL[t_name]["body"]:
            timing = time.time()  # timing
            theorem = (t_name, branch, t_para)
            gpl = self.parsed_theorem_GDL[t_name]["body"][branch]
            premises = []
            passed = True

            for predicate, item in gpl["products"] + gpl["logic_constraints"]:
                oppose = False
                if "~" in predicate:
                    oppose = True
                    predicate = predicate.replace("~", "")
                item = tuple(letters[i] for i in item)
                has_item = self.problem.condition.has(predicate, item)
                if has_item:
                    premises.append(self.problem.condition.get_id_by_predicate_and_item(predicate, item))

                if (not oppose and not has_item) or (oppose and has_item):
                    passed = False
                    break

            if not passed:
                self.problem.step(theorem, time.time() - timing)
                continue

            for equal, item in gpl["algebra_constraints"]:
                oppose = False
                if "~" in equal:
                    oppose = True
                eq = get_equation_from_tree(self.problem, item, True, letters)
                solved_eq = False

                result, premise = EqKiller.solve_target(eq, self.problem)
                if result is not None and rough_equal(result, 0):
                    solved_eq = True
                premises += premise

                if (not oppose and not solved_eq) or (oppose and solved_eq):
                    passed = False
                    break

            if not passed:
                self.problem.step(theorem, time.time() - timing)
                continue

            for predicate, item in gpl["conclusions"]:
                if predicate == "Equal":  # algebra conclusion
                    eq = get_equation_from_tree(self.problem, item, True, letters)
                    update = self.problem.add("Equation", eq, premises, theorem) or update
                else:  # logic conclusion
                    item = tuple(letters[i] for i in item)
                    update = self.problem.add(predicate, item, premises, theorem) or update

            self.problem.step(theorem, time.time() - timing)

        timing = time.time()  # timing
        EqKiller.solve_equations(self.problem)
        self.problem.step("solve_equations", time.time() - timing)

        return update

    def apply_theorem_by_name_and_branch(self, t_name, t_branch):
        """
        Apply a theorem with t_name and t_branch.
        :param t_name: <str>.
        :param t_branch: <str>.
        :return update: <bool>, Whether the question condition updated or not.
        """
        update = False

        timing = time.time()  # timing
        gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]

        conclusions = GPLExecutor.run(gpl, self.problem)  # get gpl reasoned result
        if len(conclusions) == 0:
            theorem = (t_name, t_branch, None)
            self.problem.step(theorem, time.time() - timing)
            return False
        avg_timing = (time.time() - timing) / len(conclusions)
        for letters, premise, conclusion in conclusions:
            t_para = [letters[i] for i in self.parsed_theorem_GDL[t_name]["vars"]]
            theorem = (t_name, t_branch, tuple(t_para))

            for predicate, item in conclusion:  # add conclusion
                update = self.problem.add(predicate, item, premise, theorem) or update
            self.problem.step(theorem, avg_timing)

        timing = time.time()  # timing
        EqKiller.solve_equations(self.problem)
        self.problem.step("solve_equations", time.time() - timing)

        return update

    def apply_theorem_by_name_and_para_and_branch(self, t_name, t_branch, t_para):
        """
        Apply a theorem with t_name, t_branch and t_para.
        :param t_name: <str>.
        :param t_branch: <str>.
        :param t_para: tuple of <str>.
        :return update: <bool>, Whether the problem condition updated or not.
        """
        update = False
        timing = time.time()  # timing
        theorem = (t_name, t_branch, t_para)

        letters = {}  # used for vars-letters replacement
        for i in range(len(self.parsed_theorem_GDL[t_name]["vars"])):
            letters[self.parsed_theorem_GDL[t_name]["vars"][i]] = t_para[i]

        gpl = self.parsed_theorem_GDL[t_name]["body"][t_branch]
        premises = []

        for predicate, item in gpl["products"] + gpl["logic_constraints"]:
            oppose = False
            if "~" in predicate:
                oppose = True
                predicate = predicate.replace("~", "")
            item = tuple(letters[i] for i in item)
            has_item = self.problem.condition.has(predicate, item)
            if has_item:
                premises.append(self.problem.condition.get_id_by_predicate_and_item(predicate, item))

            if (not oppose and not has_item) or (oppose and has_item):
                self.problem.step(theorem, time.time() - timing)
                return False

        for equal, item in gpl["algebra_constraints"]:
            oppose = False
            if "~" in equal:
                oppose = True
            eq = get_equation_from_tree(self.problem, item, True, letters)
            solved_eq = False

            result, premise = EqKiller.solve_target(eq, self.problem)
            if result is not None and rough_equal(result, 0):
                solved_eq = True
            premises += premise

            if (not oppose and not solved_eq) or (oppose and solved_eq):
                self.problem.step(theorem, time.time() - timing)
                return False

        for predicate, item in gpl["conclusions"]:
            if predicate == "Equal":  # algebra conclusion
                eq = get_equation_from_tree(self.problem, item, True, letters)
                update = self.problem.add("Equation", eq, premises, theorem) or update
            else:  # logic conclusion
                item = tuple(letters[i] for i in item)
                update = self.problem.add(predicate, item, premises, theorem) or update
        self.problem.step(theorem, time.time() - timing)

        timing = time.time()  # timing
        EqKiller.solve_equations(self.problem)
        self.problem.step("solve_equations", time.time() - timing)

        return update

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
                if self.problem.condition.items[_id][0] in self.ignore_ops:
                    continue
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
        
        # algebra_selections = self.try_theorem_algebra(related_syms)
        algebra_selections = []
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
        return selections
    
    def get_theorem_selection_v2(self):
        """
        Return theorem selections according to <self.last_step>.
        :return selections: <list> of ((t_name, t_branch, t_para), ((predicate, item, premise))).
        """
        selections = []

        timing = time.time()
        related_pres = []  # new added predicates
        related_syms = []  # new added/updated equations
        for step in range(0, self.problem.condition.step_count):  # get related conditions
            for _id in self.problem.condition.ids_of_step[step]:
                if self.problem.condition.items[_id][0] in self.ignore_ops:
                    continue
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
        
        # algebra_selections = self.try_theorem_algebra(related_syms)
        algebra_selections = []
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
        debug_print(self.debug, "(timing={:.4f}s) Get {}  selections: {}.".format(
            time.time() - timing, len(selections), selections))

        return selections
    
    def expand_conditions(self):
        selections = self.get_theorem_selection()
        # debug_print(selections)
        for selection in selections:
            t_msg, conclusions = selection
            for predicate, item, premise in conclusions:
                self.problem.add(predicate, item, premise, t_msg, skip_check=True)

            EqKiller.solve_equations(self.problem) 
            self.problem.step(t_msg, 0)
 