import numpy as np

from matplotlib import pyplot as plt
import string
from allocator import Allocator
import random
import re
import math
from PIL import Image, ImageDraw, ImageFont
import sympy
from sympy import Eq, Expr, Float, cos, pi, simplify, solve, symbols

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils import (PREDICATES_ENT, PREDICATES_REL, find_target_for_construct,
                   get_content, get_points, get_predicate_name, get_symbol,
                   max_letter_index, parse_clause, replace_points, setup_seed)
import cv2


class Plotter():
    def __init__(self, allocater_states, min_side=250, max_side=350, allocate_char_mode='order'):
        self.p_pos = allocater_states['p_pos']
        self.lines = allocater_states['lines']
        self.circles = allocater_states['circles']
        self.points_on_circle = allocater_states['points_on_circle']
        self.min_side = min_side
        self.max_side = max_side
        
        fig_size_x, fig_size_y = self.normalize_positions()
        # numpy image: (y, x, channel)
        self.fig = np.zeros((fig_size_y, fig_size_x, 3), np.uint8)
        self.fig.fill(255)
        # image = Image.new('RGB', (fig_size_x, fig_size_y), 'white')
        # self.fig = ImageDraw.Draw(image)
        
        # Color: BGR
        self.l_color = (0, 0, 0)
        self.p_color = (151, 85, 47)
        self.c_color = (204, 72, 6)
        self.line_width = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        
        assert allocate_char_mode in ['random', 'order']
        self.allocate_char_mode = allocate_char_mode
        
    @property
    def radius(self):
        c = self.circles[0]
        p0 = self.points_on_circle[self.circles[0]][0]
        xc, yc = self.p_pos[c]
        x1, y1 = self.p_pos[p0]
        return int(self.distance([xc, yc], [x1, y1]))
        
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            
    def normalize_positions(self, pad_size=50):
        # normalize positions for points, add padding and return fig size
        points = [p for p, pos in self.p_pos.items()]
        positions = list(self.p_pos.values())
        min_x = np.min([pos[0] for pos in positions])
        max_x = np.max([pos[0] for pos in positions])
        min_y = np.min([pos[1] for pos in positions])
        max_y = np.max([pos[1] for pos in positions])
        
        # check if circle out of figure
        if len(self.circles) > 0:
            circle_x, circle_y = self.p_pos[self.circles[0]]
            radius = self.radius
            min_x = min(min_x, circle_x - radius)
            max_x = max(max_x, circle_x + radius)
            min_y = min(min_y, circle_y - radius)
            max_y = max(max_y, circle_y + radius)
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # scale according to length of min line of bbox
        short_side = min(bbox_width, bbox_height)
        scale = random.uniform(self.min_side, self.max_side) / short_side
        # shift, scale for points
        p_pos_norm = {}
        for p, pos in self.p_pos.items():
            p_pos_norm[p] = [
                int((pos[0] - min_x) * scale),
                int((pos[1] - min_y) * scale)
            ]
        self.p_pos = p_pos_norm
        
        # pad for points
        p_pos_norm = {}
        for p, pos in self.p_pos.items():
            p_pos_norm[p] = [pos[0] + pad_size, pos[1] + pad_size]
        self.p_pos = p_pos_norm
        
        fig_size_x = (max_x - min_x) * scale + 2 * pad_size
        fig_size_y = (max_y - min_y) * scale + 2 * pad_size
                
        return int(fig_size_x), int(fig_size_y)
        
    def get_best_chars_position(self, text_size, point_pos):
        # 返回的position -> 文字bbox的左下角
        radiuses = [10, 15, 20]
        position_pixel_vals = []
        position_to_select = []
        for radius in radiuses:
            # 文字中心分布在离点radius远的圆周上
            for _ in range(10):
                theta = math.radians(random.uniform(0, 360))
                x = point_pos[0] + math.cos(theta) * radius
                y = point_pos[1] + math.sin(theta) * radius
                char_bbox = [max(x - text_size[0] / 2, 5), max(y - text_size[1] / 2, 5),
                             x + text_size[0] / 2, y + text_size[1] / 2]

                char_bbox = [int(x) for x in char_bbox]
                # 不与图像交叠，字母bbox中的像素值全为255，(y, x, channel)
                img_reserve = 255 - self.fig[char_bbox[1]:char_bbox[3],
                                             char_bbox[0]:char_bbox[2], :]
                pixel_val = sum(img_reserve.reshape(-1)) 
                sum(255 - self.fig[char_bbox[0]:char_bbox[2], char_bbox[1]:char_bbox[3], :])
                if abs(pixel_val) < 1e-5:
                    # cv2.circle(self.fig, (int(x), int(y)), 4, color=self.p_color, thickness=-1, lineType=cv2.LINE_AA)
                    # cv2.rectangle(self.fig, [char_bbox[0], char_bbox[1]], [char_bbox[2], char_bbox[3]],(255, 0, 0), 1)
                    # shift
                    res_pos = [int(max(x - text_size[0] / 2, 5)),  int(y + text_size[1] / 2)]
                    # return [char_bbox[0], char_bbox[1]]
                    return res_pos
                
                position_pixel_vals.append(pixel_val)
                position_to_select.append([int(max(x - text_size[0] / 2, 5)),  int(y + text_size[1] / 2)])
        
        # 如果都有交叠，选择sum最小的位置
        idx = np.argmin(position_pixel_vals)
        return position_to_select[idx]
        
    def plot(self):
        # plot points
        for p, pos in self.p_pos.items():
            cv2.circle(self.fig, pos, 4, color=self.p_color, thickness=-1, lineType=cv2.LINE_AA)
            # self.fig.ellipse()
            
        # plot lines
        for line in self.lines:
            for i in range(len(line)-1):
                p1_pos, p2_pos = self.p_pos[line[i]], self.p_pos[line[i+1]]
                cv2.line(self.fig, p1_pos, p2_pos, self.l_color, self.line_width, lineType=cv2.LINE_AA)
            
        # plot circles
        for circle in self.circles:
            c_pos = self.p_pos[circle]
            cv2.circle(self.fig, c_pos, self.radius, color=self.l_color, 
                       thickness=self.line_width, lineType=cv2.LINE_AA)
            
        # plot chars
        if self.allocate_char_mode == 'random':
            chars = random.sample(string.ascii_uppercase, len(self.p_pos))
        else:
            chars = list(string.ascii_uppercase[:len(self.p_pos)])
        point_mapping = {}
        for point_i, char in zip(self.p_pos, chars):
            point_mapping[point_i] = char
            # char, font, size, width
            text_width = 2
            text_size = 1
            text_bbox_size, _ = cv2.getTextSize(char, self.font, text_size, text_width)
            # text pos is the top left pos
            text_pos = self.get_best_chars_position(text_bbox_size, self.p_pos[point_i])
            cv2.putText(self.fig, char, text_pos, self.font, text_size, 
                        self.c_color, text_width, cv2.LINE_AA)
            
    def save_fig(self, fig_name, fig_dir):
        cv2.imwrite(f"{fig_dir}/{fig_name}.png", self.fig)

if __name__ == '__main__':
    setup_seed(124)
    dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    for i in range(5):
        # clauses_base = random.choices(PREDICATES_ENT, k=1)
        # clauses_rel = random.choices(PREDICATES_REL, k=2)
        clauses_base = [
            # 'EquilateralTriangle'
            # 'RightTriangle',
            # "Triangle",
            # "Parallelogram",
            # "Rectangle",
            # "Rhombus",
            # "RightTriangle",
            "Square",
            # "EquilateralTriangle",
            # "IsoscelesTriangle",
            # "Trapezoid",
            # "Kite",
            # "RightTrapezoid",
            # "IsoscelesTrapezoid",
            # "IsoscelesRightTriangle",
            
            # "SimilarBetweenTriangle",
            # "SimilarBetweenQuadrilateral",
            # "CongruentBetweenTriangle",
            # "CongruentBetweenQuadrilateral",
        ]
        clauses_rel = [
            'IsIncenterOfTriangle', 
            'IsMidpointOfArc',
            # 'CongruentBetweenArc'
            ]
        
        cg = ClauseGenerator(dl.predicate_GDL, dl.theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_new_lines=0
        )
        print('---------- Chosen Predicates ----------')
        print('clauses_base: ', clauses_base)
        print('clauses_rel: ', clauses_rel)
        
        allocator = Allocator(cg.states, c_cdls, t_cdls, dl.predicate_GDL)
        print('---------- Formulated CDLs ----------')
        formulated_cdls = allocator.formulated_cdls
        print('Text CDLs: ')
        for t_cdl in formulated_cdls['text_cdls']:
            print('\t', t_cdl)
        print('Construct CDLs: ')
        for c_cdl in formulated_cdls['construct_cdls']:
            print('\t', c_cdl)
            
        allocator.allocate()
        
        
        
        print("---------- Location ----------")
        for p, pos in allocator.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
            
        plotter = Plotter(allocator.states)
        plotter.plot()
        print("---------- Location Norm ----------")
        for p, pos in plotter.p_pos.items():
            print(f"{p}: [{pos[0]:.3f}, {pos[1]:.3f}]")
        plotter.save_fig('test', 'imgs_test')
        
        print('==============================================')
