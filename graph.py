import json
import random
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from tqdm import tqdm
from sympy import Eq

from formalgeo.data import (DatasetLoader, download_dataset,
                            show_available_datasets)
from formalgeo.parse import parse_theorem_seqs
from formalgeo.problem.condition import Goal
from formalgeo.solver import Interactor
from formalgeo.tools import show_solution
from utils import move_subtractions_to_rhs
# show_available_datasets()

class ConditionNode():
    def __init__(self, value):
        self.value = value
        if type(self.value[1]) == tuple:
            self.v_str = f"{self.value[0]}\\n{''.join(self.value[1])}"
        else:
            self.v_str = f"{self.value[0]}\\n{str(self.value[1])}"
    
    def __repr__(self) -> str:
        if type(self.value[1]) == tuple:
            return f"{self.value[0]}({','.join(self.value[1])}) | {self.value[2]}"
        else:
            return f"{self.value[0]}({str(self.value[1])}) | {self.value[2]}"
        
            
class ConditionGraph():
    def __init__(self, conditions: List[Tuple]):
        self.conditions: List[Tuple] = conditions
        # All nodes
        self.nodes: List[ConditionNode] = []
        self.nodes_adj_table: Dict[ConditionNode, List[ConditionNode]] = {}
        
        # cache all sub_graph, k: (end_nodes) v: (nodes, nodes_adj_table)
        self.sub_graph_dict: Dict[List[ConditionNode], Tuple] = {}

        self.ignore_ops = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle', 'Arc']
        self.end_nodes = None
        
    def construct_graph(self):
        for condition in self.conditions:
            node = ConditionNode(condition)
            if node not in self.nodes:
                self.nodes.append(node)

                # self.nodes[idx] -> node
                for idx in node.value[2]:
                    if idx == -1: break 
                    if self.nodes[idx] not in self.nodes_adj_table:
                        self.nodes_adj_table[self.nodes[idx]] = [node]
                    else:
                        self.nodes_adj_table[self.nodes[idx]] += [node]
                        
                        
    def get_end_nodes(self):
        if self.end_nodes is None or len(self.end_nodes) == 0:
            # find all end nodes
            end_nodes = []
            for node in self.nodes:
                if node in self.nodes_adj_table:
                    if len(self.nodes_adj_table[node]) == 0:
                        end_nodes.append(node)
                    else:
                        continue
                else:
                    end_nodes.append(node)
            
            # filter 
            end_nodes_to_save = []
            operatior_to_delete = ['Angle', 'Line', 'Point', 'Shape', 'Polygon', 'Triangle']
            for end_node in end_nodes:
                operator = end_node.value[0]
                if not any([operator==s for s in operatior_to_delete]):
                    if end_node.value[-1] != 0:
                        end_nodes_to_save.append(end_node)
            self.end_nodes = end_nodes_to_save
        return self.end_nodes
    
    def get_end_conditions(self):
        end_conditions = [n.value for n in self.get_end_nodes()]
        return end_conditions
    
    def calculate_depth(self, goal: List[Union[Tuple, ConditionNode]]):
        if type(goal) == ConditionNode:
            goal = goal.value
        if goal[2][0] == -1: # from origin
            return 0
        if goal[-1] == 0 and goal[0] in self.ignore_ops: # from direct extend
            return 0
        parents = [self.conditions[idx] for idx in goal[2]]
        depths = [self.calculate_depth(p) for p in parents]
        max_depth = 1 + max(depths)
        
        return max_depth
            
        
                        
    def backward_construct_sub_graph(
        self, 
        goals: List[Union[Tuple, ConditionNode]]
    )->Tuple[List[ConditionNode], Dict[ConditionNode, List[ConditionNode]]]:
        """construct sub graph accroding to end nodes (goals). """
        
        sub_nodes: List[ConditionNode] = []
        sub_nodes_adj_table: Dict[ConditionNode, List[ConditionNode]] = {}
        goal_nodes = []
        
        # Tuple -> Node
        for goal in goals:
            if type(goal) == tuple:
                goal_idx = self.conditions.index(goal)
                goal = self.nodes[goal_idx]
            goal_nodes.append(goal)
        goal_nodes = tuple(goal_nodes)
        
        def add_nodes(node: ConditionNode):
            """recursive add parent nodes"""
            if node not in sub_nodes:
                sub_nodes.append(node)
            
            # self.nodes[idx] -> node
            for idx in node.value[2]: # Recursion end at this
                if idx == -1:
                    break 
                if node.value[-1] == 0 and node.value[0] in self.ignore_ops:
                    break
                add_nodes(self.nodes[idx])
                
                if self.nodes[idx] not in sub_nodes_adj_table:
                    sub_nodes_adj_table[self.nodes[idx]] = [node]
                else:
                    sub_nodes_adj_table[self.nodes[idx]].append(node)
        # if not cached
        if goal_nodes not in self.sub_graph_dict:
            for goal_node in goal_nodes:
                assert type(goal_node) == ConditionNode
                if goal_node not in sub_nodes:
                    sub_nodes.append(goal_node)
                add_nodes(goal_node)
            self.sub_graph_dict[goal_nodes] = (sub_nodes, sub_nodes_adj_table)
            
        return self.sub_graph_dict[goal_nodes]
    
    def backward_delete_sub_graph(self, goal: Tuple):
        """delete nodes that have paths ended with goal node (strict)"""
        goal_idx = self.conditions.index(goal)
        goal_node = self.nodes[goal_idx]
        
        def delete_nodes(node: ConditionNode):
            """recursive delete parant nodes"""
            self.sub_nodes.remove(node)
            parent_nodes = [self.nodes[idx] for idx in node.value[2]]
            for p_node in parent_nodes:
                self.sub_nodes_adj_table[p_node].remove(node)
            
            for p_node in parent_nodes:
                res = self.check_parent_strict(p_node, goal_node)
                if res:
                    delete_nodes(p_node)
        
        delete_nodes(goal_node)
        return self.sub_nodes_adj_table
    
    def check_parent(self, node: ConditionNode, end_node: ConditionNode):
        """只要有一条经过node的路径以end_node结束，就认为node是end_node的parent"""
        assert node in self.nodes and end_node in self.nodes
        if node == end_node: 
            return True
        if node not in self.nodes_adj_table: # node is also an end point
            return False
        if end_node in self.nodes_adj_table[node]: # node -> end_node
            return True
        
        res_list = []
        for next_node in self.nodes_adj_table[node]:
            res = self.check_parent(next_node, end_node)
            res_list.append(res)
            
        return any(res_list)
    
    def check_parent_strict(self, node: ConditionNode, end_node: ConditionNode):
        """所有经过node的路径都以end_node结束，才认为node是end_node的parent"""
        assert node in self.nodes and end_node in self.nodes
        if node == end_node:
            return True
        if node not in self.nodes_adj_table: # node is also an end point
            return False
        if end_node in self.nodes_adj_table[node] and len(self.nodes_adj_table[node]) == 1:
            return True
        
        res_list = []
        for next_node in self.nodes_adj_table[node]:
            res = self.check_parent_strict(next_node, end_node)
            res_list.append(res)
            
        return all(res_list)
    
    def visualize(self, 
                  node_list: List[ConditionNode],
                  adj_table: Dict[ConditionNode, List[ConditionNode]],
                  fig_name="test",
                  img_dir="imgs"):
        
        G = nx.DiGraph()  # 创建一个有向图    
        for node in node_list:
            G.add_node(node.v_str, label=node.v_str)
            if node in adj_table:
                for next_node in adj_table[node]:
                    G.add_edge(node.v_str, next_node.v_str)

        # pos = nx.spring_layout(G)  # 弹簧布局
        # pos = nx.circular_layout(G) # 圆形布局
        # pos = nx.spectral_layout(G) # 谱布局


        pos = graphviz_layout(G, prog='dot')  # dot布局
        # pos = graphviz_layout(G, prog='circo')  
        # pos = graphviz_layout(G, prog='twopi') 
        # pos = graphviz_layout(G, prog='neato') 
         

        labels = {n: n.replace('\\n', '\n') for n in G.nodes()} 
       
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, font_size=6, font_color='black', node_color='skyblue')
        plt.title(f'Condition Graph Vis of {fig_name}')
        
        fig_name = f"graph_{fig_name}"
        plt.savefig(f"{img_dir}/{fig_name}.png", dpi=1000)
        plt.clf()
        
    
def draw_graph(condition_graph, idx, target_condition=None, img_dir="imgs"):
    
    if target_condition is not None:
        sub_nodes, sub_nodes_adj_table = condition_graph.backward_construct_sub_graph([target_condition])
        goal_syms = list(target_condition[1].free_symbols)
        if len(goal_syms) == 1:
            goal_sym = goal_syms[0]
        else:
            goal_sym = str(target_condition[1]).replace(' ', '')
        condition_graph.visualize(
            node_list=sub_nodes,
            adj_table=sub_nodes_adj_table,
            fig_name=f"{idx}_goal_{str(goal_sym)}",
            img_dir=img_dir
        )
    else:
        end_nodes = condition_graph.get_end_nodes()
        nodes, nodes_adj_table = condition_graph.backward_construct_sub_graph(end_nodes)
        condition_graph.visualize(
            node_list=nodes,
            adj_table=nodes_adj_table,
            fig_name=f"{idx}_total",
            img_dir=img_dir
        )
    
def topological_sort(nodes, nodes_adj_table):
    def dfs(node, visited, stack):
        visited.add(node)
        for neighbor in nodes_adj_table.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, visited, stack)
        stack.append(node)
    
    visited = set()
    stack = []
    
    for node in nodes:
        if node not in visited:
            dfs(node, visited, stack)
    
    # Since the stack is built in reverse order, we need to reverse it before returning
    return stack[::-1]


    
def display_solution(condition_graph: ConditionGraph, 
                     target_condition: Tuple,
                     goal: Goal):
    sub_nodes, sub_nodes_adj_table = condition_graph.backward_construct_sub_graph([target_condition])
    
    # sort by index of applying theorem 
    sub_nodes = topological_sort(sub_nodes, sub_nodes_adj_table)

    solutions = []
    solution_dict = {}
    for i, node in enumerate(sub_nodes):
        theorem = node.value[3][0]
        statement = str(node).split('|')[0].strip()
        if 'Equation' in statement:
            pattern = r"Equation\((.*?)\)"
            matches = re.findall(pattern, statement)[0]
            statement = f"{matches} = 0"
        if node.value[-1] == 0:
            solution_i = f"{theorem}: {statement} [{i}]"
            pre_idx = [-1]
        else:
            pre_nodes = [condition_graph.nodes[j] for j in node.value[2]]
            pre_idx = [sub_nodes.index(n) for n in pre_nodes]
            pre_idx_str = "".join([f"[{x}]" for x in pre_idx])
            solution_i = f"{theorem}: {pre_idx_str} => {statement} {[i]}"
        solutions.append(solution_i)
        solution_dict[i] = {
            "theorem": theorem,
            "condition": str(pre_idx),
            "statement": statement
        }
    
    solution_str = "Solution: \n" + "\n".join(solutions)
    final_str = f"\nSolved: {goal.solved}\n"
    if goal.solved and goal.solved_answer != None:
        if goal.answer == 0: # expression     
            eq_str = move_subtractions_to_rhs(Eq(goal.item, 0))
            final_str += f"Final Answer: {eq_str}" 
        else: 
            final_str += f"Final Answer: {goal.item} = {goal.solved_answer}"
    else:
        final_str += f"Final Answer: None"
    solution_str += final_str
    
    info_dict = {
        "solved": goal.solved,
        "item": str(goal.item), 
        "target_answer": str(goal.answer),
        "solved_answer": str(goal.solved_answer),
        "solutions": solution_dict,
        "solution_str": solution_str
    }
    
    # print(solution_str)
    return info_dict


