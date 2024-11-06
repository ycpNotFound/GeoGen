import numpy as np

from matplotlib import pyplot as plt
import string
from allocator import Allocator
import random
import re
import math
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sympy
from sympy import Eq, Expr, Float, cos, pi, simplify, solve, symbols

from formalgeo.data import DatasetLoader
from generator import ClauseGenerator
from utils import (PREDICATES_ENT, PREDICATES_REL, PREDICATES_REL_2,  PRESET_COLORS, PREDICATES_TO_NAMES, find_target_for_construct,
                   get_content, get_points, get_predicate_name, get_symbol,
                   max_letter_index, parse_clause, replace_points, setup_seed, replace_for_clause, clause_to_nature_language, convert_upper_to_lower)
import cv2
import networkx as nx
from utils import hex_to_bgr


class Plotter():
    def __init__(self, 
                 geo_states, 
                 text_cdls,
                 construct_cdls, 
                 image_cdls,
                 min_side=300, 
                 max_side=350, 
                 color_config=PRESET_COLORS[0],
                 replace_characters=False,
                 debug=False):
        self.p_pos = geo_states['p_pos']
        self.lines = geo_states['lines']
        self.circles = geo_states['circles']
        self.points_on_circle = geo_states['points_on_circle']
        
        self.text_cdls = text_cdls
        self.construct_cdls = construct_cdls
        self.image_cdls = image_cdls
        self.debug = debug
        
        self.min_side = min_side
        self.max_side = max_side
        
        if replace_characters:
            self.replace_characters()
        
        fig_size_x, fig_size_y = self.normalize_positions()
        self.fig_size = (fig_size_x, fig_size_y)
        # numpy image: (y, x, channel)
        self.fig = np.zeros((fig_size_y, fig_size_x, 3), np.uint8)
        self.fig.fill(255)
        # image = Image.new('RGB', (fig_size_x, fig_size_y), 'white')
        # self.fig = ImageDraw.Draw(image)
        
        # Color: BGR
        self.l_color = hex_to_bgr(color_config['l_color'])    # line
        self.p_color = hex_to_bgr(color_config['p_color'])    # point
        self.c_color = hex_to_bgr(color_config['c_color'])    # char
        self.a_color = hex_to_bgr(color_config['a_color'])    # annotation
        self.f_color = hex_to_bgr(color_config['f_color'])    # fill_in
        self.line_width = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # draw annotations: perp, equal, fill_in_color entity ..
        self.annotation_targets = self.find_annotation_clauses()
        font_size = int(min(27, min(self.fig_size) / 15))
        self.FT = ImageFont.truetype("font/arial.ttf", font_size)
        if self.debug:
            print("Font Size: ", font_size)
            
        natural_template_path = "json/natural_language_template.json"
        self.natural_template = json.load(open(natural_template_path, 'r'))
        
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    

    def replace_characters(self):
        if self.debug:
            start_char = 'A'
        else:
            start_char = random.choice(['A', 'E', 'I', 'M', 'R'])
        start_idx = string.ascii_uppercase.index(start_char)
        if start_idx + len(self.p_pos) > len(string.ascii_uppercase):
            start_idx = len(string.ascii_uppercase) - len(self.p_pos) - 1
        chars = string.ascii_uppercase[start_idx: start_idx + len(self.p_pos)]
        mapping = {}
        for point_i, char in zip(self.p_pos, chars):
            mapping[point_i] = char
            
        # points
        self.p_pos = {mapping[k]: self.p_pos[k] for k in self.p_pos}
        # lines / circles
        self.lines = [tuple([mapping[p] for p in l]) for l in self.lines]
        self.circles = [mapping[p] for p in self.circles]
        self.points_on_circle = {mapping[k]: [mapping[x] for x in v] 
                                 for k, v in self.points_on_circle.items()}
        # clauses
        self.text_cdls = [replace_for_clause(c, mapping) for c in self.text_cdls]
        self.construct_cdls = [replace_for_clause(c, mapping) for c in self.construct_cdls]
        self.image_cdls = [replace_for_clause(c, mapping) for c in self.image_cdls]
        return
    
    def get_radius(self, c):
        p0 = self.points_on_circle[c][0]
        xc, yc = self.p_pos[c]
        x1, y1 = self.p_pos[p0]
        return self.distance([xc, yc], [x1, y1])
            
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
            for circle in self.circles:
                circle_x, circle_y = self.p_pos[circle]
                radius = self.get_radius(circle)
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
            for _ in range(20):
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
    
    def find_annotation_clauses(self):
        right_angles, eq_lines, eq_angles, eq_arcs = [], [], [], []
        length_of_line, measure_of_angle = [], []
        for clause in self.image_cdls[:]:
            if 'Equal' in clause:
                _, items = parse_clause(clause)
                if 'LengthOfLine' in clause:
                    if clause.count('LengthOfLine') == 2: 
                        # annotate eq lines
                        if len(items[0]) != 2 or len(items[1]) != 2:
                            self.image_cdls.remove(clause)
                            continue
                        eq_lines.append([tuple(items[0]), tuple(items[1])])
                    else: 
                        # annotate length for line
                        if len(items[0]) != 2:
                            self.image_cdls.remove(clause)
                        length_of_line.append([tuple(items[0]), items[1]])
                    
                if 'MeasureOfAngle' in clause:
                    if clause.count('MeasureOfAngle') == 1: 
                        if len(items[0]) != 3:
                            self.image_cdls.remove(clause)
                            continue
                        if '90' in clause: # annotate perpendicular
                            right_angles.append(tuple(items[0]))
                        else: # annotate measure for angle
                            measure_of_angle.append([tuple(items[0]), items[1]])
                    elif clause.count('MeasureOfAngle') == 2: 
                        # annotate eq angles
                        if len(items[0]) != 3 or len(items[1]) != 3:
                            self.image_cdls.remove(clause)
                            continue
                        eq_angles.append([tuple(items[0]), tuple(items[1])])
                if 'LengthOfArc' in clause:
                    if clause.count('LengthOfArc') == 2:
                        if len(items[0]) != 3 or len(items[1]) != 3:
                            self.image_cdls.remove(clause)
                            continue
                        eq_arcs.append([tuple(items[0]), tuple(items[1])])
                        
        # find AB = BC = CA 
        G_eq_line = nx.Graph()
        for eq_line in eq_lines:
            line_1, line_2 = eq_line
            line_1, line_2 = tuple(sorted(line_1)), tuple(sorted(line_2))
            G_eq_line.add_edge(line_1, line_2)
            
        eq_lines = [list(group) for group in nx.connected_components(G_eq_line)]
        
        # when AB \perp CBD, there's angle ABC, ABD = 90, delete one
        right_angles_ = []
        for i in range(len(right_angles)):
            angle_i = right_angles[i]
            repeated = False
            for j in range(i):
                angle_j = right_angles[j]
                if len(set(angle_i) & set(angle_j)) == 2:
                    if angle_i[1] == angle_j[1]:
                        repeated = True
            if not repeated:
                right_angles_.append(angle_i)
        right_angles = right_angles_
        
        # find entity to fill in color, choose quad and circle first
        entities = []
        for clause in self.construct_cdls:
            pred_name, items = parse_clause(clause)
            if pred_name == "Shape":
                if len(items) == 4:
                    entities.append(['Polygon', items])
            if pred_name == 'Cocircular':
                entities.append(['Circle', items])
                
        # if there's no quad and circle, choose triangle
        if len(entities) == 0:
            for clause in self.construct_cdls:
                pred_name, items = parse_clause(clause)
                if pred_name == "Shape":
                    entities.append(['Polygon', items])
                    
        # if there're 2 triangles / quads similar / congruent
        if 'Congruent' in self.text_cdls[0] or 'Similar' in self.text_cdls[0]:
            _, items = parse_clause(self.text_cdls[0])
            filled_entity = ['TwoPolygon', items]
        else:
            filled_entity = random.choice(entities)
            
        return {
            "right_angles": right_angles,
            "eq_lines": eq_lines,
            "eq_angles": eq_angles,
            "eq_arcs": eq_arcs,
            "filled_entity": filled_entity,
            "length_of_line": length_of_line,
            "measure_of_angle": measure_of_angle
        }
    
    def plot_right_angle(self, right_angles):
        # plot 'L' for right angle
        for right_angle in right_angles:
            # right angle ABC
            l_len = 15
            xa, ya = self.p_pos[right_angle[0]]
            xb, yb = self.p_pos[right_angle[1]]
            xc, yc = self.p_pos[right_angle[2]]
            vec_ba_len = self.distance((xa, ya), (xb, yb))
            vec_ba = ((xa - xb) / vec_ba_len, (ya - yb) / vec_ba_len)
            vec_bc_len = self.distance((xc, yc), (xb, yb))
            vec_bc = ((xc - xb) / vec_bc_len, (yc - yb) / vec_bc_len)
            # p1 in AB, p2 in AC, p3: p1 + p2
            x1, y1 = xb + vec_ba[0] * l_len, yb + vec_ba[1] * l_len
            x2, y2 = xb + vec_bc[0] * l_len, yb + vec_bc[1] * l_len
            x3 = xb + vec_ba[0] * l_len + vec_bc[0] * l_len
            y3 = yb + vec_ba[1] * l_len + vec_bc[1] * l_len
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            x3, y3 = int(x3), int(y3)
            # line: p1-p3, p2-p3
            cv2.line(self.fig, (x1, y1), (x3, y3), self.a_color, self.line_width, lineType=cv2.LINE_AA)
            cv2.line(self.fig, (x2, y2), (x3, y3), self.a_color, self.line_width, lineType=cv2.LINE_AA)
            
    def plot_equal_lines(self, eq_lines):
        # plot 1/2/3 short perpendicular segment for equal lines
        for idx, eq_line in enumerate(eq_lines):
            l_len = 8
            for l_i in eq_line: # line AB
                xa, ya = self.p_pos[l_i[0]]
                xb, yb = self.p_pos[l_i[1]]
                xm, ym = (xa + xb) / 2, (ya + yb) / 2
                ab_len = self.distance((xa, ya), (xb, yb))
                ab_vec = ((xa - xb) / ab_len, (ya - yb) / ab_len)
                ab_perp = (-ab_vec[1], ab_vec[0])
                if idx == 0: # draw 1 perp line
                    mid_ps = [[xm, ym]]
                elif idx == 1: # draw 2 perp line
                    xm_1, ym_1 = xm - 5 * ab_vec[0], ym - 5 * ab_vec[1]
                    xm_2, ym_2 = xm + 5 * ab_vec[0], ym + 5 * ab_vec[1]
                    mid_ps = [[xm_1, ym_1], [xm_2, ym_2]]
                elif idx == 2: # draw 3 perp line
                    xm_1, ym_1 = xm - 10 * ab_vec[0], ym - 10 * ab_vec[1]
                    xm_2, ym_2 = xm + 10 * ab_vec[0], ym + 10 * ab_vec[1]
                    mid_ps = [[xm_1, ym_1], [xm, ym], [xm_2, ym_2]]
                else:
                    continue
                
                for mid_p in mid_ps:
                    xm_i, ym_i = mid_p
                    x1, y1 = xm_i + ab_perp[0] * l_len, ym_i + ab_perp[1] * l_len
                    x2, y2 = xm_i - ab_perp[0] * l_len, ym_i - ab_perp[1] * l_len
                    cv2.line(self.fig, (int(x1), int(y1)), (int(x2), int(y2)), self.a_color, self.line_width, lineType=cv2.LINE_AA)
        
    def plot_equal_anlges(self, eq_angles):    
        for idx, eq_angle in enumerate(eq_angles):
            angle_1, angle_2 = eq_angle
            # angle ABC
            xa, ya = self.p_pos[angle_1[0]]
            xb, yb = self.p_pos[angle_1[1]]
            xc, yc = self.p_pos[angle_1[2]]
            line_len = min(self.distance([xa, ya], [xb, yb]),
                           self.distance([xc, yc], [xb, yb]))
            radius = min([30, int(line_len / 4)])
            if radius <= 8:
                continue
            # ensure BA -> BC clock wise, BA x BC < 0 
            # for cv2, ensure BA -> BC counter clock wise, BA x BC > 0 
            # BA = [xa - xb, ya - yb]
            # BC = [xc - xb, yc - yb]
            if (xa - xb) * (yc - yb) - (ya - yb) * (xc - xb) < 0:
                xa, ya, xc, yc = xc, yc, xa, ya
            angle_BC = np.degrees(np.arctan2(yc-yb, xc-xb))
            angle_BA = np.degrees(np.arctan2(ya-yb, xa-xb))
            angle_BC = angle_BC + 360 if angle_BC < 0 else angle_BC
            angle_BA = angle_BA + 360 if angle_BA < 0 else angle_BA
            if angle_BC < angle_BA:
                angle_BA = angle_BA - 360
            # print(angle_BA, angle_BC)
            # B as circle center
            cv2.ellipse(self.fig, (xb, yb), (radius, radius), 0, angle_BA, angle_BC, self.a_color, self.line_width, lineType=cv2.LINE_AA)
            
            # angle ABC
            xa, ya = self.p_pos[angle_2[0]]
            xb, yb = self.p_pos[angle_2[1]]
            xc, yc = self.p_pos[angle_2[2]]
            # ensure BA -> BC clock wise, BA x BC < 0 
            # for cv2, ensure BA -> BC counter clock wise, BA x BC > 0 
            # BA = [xa - xb, ya - yb]
            # BC = [xc - xb, yc - yb]
            if (xa - xb) * (yc - yb) - (ya - yb) * (xc - xb) < 0:
                xa, ya, xc, yc = xc, yc, xa, ya
            angle_BC = np.degrees(np.arctan2(yc-yb, xc-xb))
            angle_BA = np.degrees(np.arctan2(ya-yb, xa-xb))
            angle_BC = angle_BC + 360 if angle_BC < 0 else angle_BC
            angle_BA = angle_BA + 360 if angle_BA < 0 else angle_BA
            if angle_BC < angle_BA:
                angle_BA = angle_BA - 360
            # print(angle_BA, angle_BC)
            # B as circle center, radius -= delta to distinguish with the other
            delta_r = min(5, radius // 2)
            cv2.ellipse(self.fig, (xb, yb), (radius-delta_r, radius-delta_r), 0, angle_BA, angle_BC, self.a_color, self.line_width, lineType=cv2.LINE_AA)
    
    def plot_equal_arcs(self, eq_arcs):
        for idx, eq_arc in enumerate(eq_arcs):
            l_len = 8
            for arc_i in eq_arc:
                # acr OAB
                xo, yo = self.p_pos[arc_i[0]]
                xa, ya = self.p_pos[arc_i[1]]
                xb, yb = self.p_pos[arc_i[2]]
                r_len = self.distance((xo, yo), (xa, ya))
                # OA + OB
                mid_vec = (xa - xo + xb - xo, ya - yo + yb - yo)
                mid_vec_len = math.sqrt(mid_vec[0]**2 + mid_vec[1]**2)
                mid_vec_unit = (mid_vec[0] / mid_vec_len, mid_vec[1] / mid_vec_len)
                xm, ym = xo + r_len * mid_vec_unit[0], yo + r_len * mid_vec_unit[1]
                
                if idx == 0: # draw 1 perp line
                    mid_ps = [[xm, ym]]
                elif idx == 1: # draw 2 perp line
                    tangent_vec = - mid_vec_unit[1], mid_vec_unit[0]
                    xm_1, ym_1 = xm - 3 * tangent_vec[0], ym - 3 * tangent_vec[1]
                    xm_2, ym_2 = xm + 3 * tangent_vec[0], ym + 3 * tangent_vec[1]
                    mid_ps = [[xm_1, ym_1], [xm_2, ym_2]]
                else:
                    continue
                for mid_p in mid_ps:
                    xm_i, ym_i = mid_p
                    x1, y1 = xm_i + mid_vec_unit[0] * l_len, ym_i + mid_vec_unit[1] * l_len
                    x2, y2 = xm_i - mid_vec_unit[0] * l_len, ym_i - mid_vec_unit[1] * l_len
                    cv2.line(self.fig, (int(x1), int(y1)), (int(x2), int(y2)), self.a_color, self.line_width, lineType=cv2.LINE_AA)
            
    def plot_fill_in_color(self, filled_entity):
        ent_type, items = filled_entity
        if ent_type == 'Polygon':
            points = [i[0] for i in items]
            pos = np.array([self.p_pos[p] for p in points])
            pos = pos.reshape((-1, 1, 2))  
            cv2.fillPoly(self.fig, [pos], self.f_color)
        elif ent_type == 'Circle':
            center_pos = self.p_pos[items[0]]
            radius = self.get_radius(items[0])
            cv2.circle(self.fig, center_pos, int(radius), self.f_color, -1)
        elif ent_type == 'TwoPolygon':
            pos_1 = np.array([self.p_pos[p] for p in items[0]])
            pos_1 = pos_1.reshape((-1, 1, 2))
            pos_2 = np.array([self.p_pos[p] for p in items[1]])
            pos_2 = pos_2.reshape((-1, 1, 2))
            
            if self.f_color != hex_to_bgr("#ffffff"):
                cv2.fillPoly(self.fig, [pos_1], hex_to_bgr("#9DC7DD")) # blue
                cv2.fillPoly(self.fig, [pos_2], hex_to_bgr("#B5D69E")) # green
            
            else:
                cv2.fillPoly(self.fig, [pos_1], self.f_color) # blue
                cv2.fillPoly(self.fig, [pos_2], self.f_color) # green
            
        else:
            raise KeyError(ent_type)
        
    def plot_length_of_line(self, length_of_line):
        for line, length in length_of_line:
            xa, ya = self.p_pos[line[0]]
            xb, yb = self.p_pos[line[1]]
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            
            BA_length = self.distance([xb, yb], [xa, ya])
            BA_unit = ((xa - xb) / BA_length, (ya - yb) / BA_length)
            BA_perp_unit = (-BA_unit[1], BA_unit[0])
            # 文字中心在离心方向, (O->m) * BA_perp_unit > 0, add, else minus
            xo, yo = self.fig_size[0] / 2, self.fig_size[1] / 2
            vec_1 = (xm - xo) * BA_perp_unit[0] + (ym - yo) * BA_perp_unit[1]
            if vec_1 > 0:
                xc, yc = xm + BA_perp_unit[0] * 35, ym + BA_perp_unit[1] * 35
            else:
                xc, yc = xm - BA_perp_unit[0] * 35, ym - BA_perp_unit[1] * 35
            
            # cv2.circle(self.fig, (int(xc), int(yc)), 4, color=self.p_color, thickness=-1, lineType=cv2.LINE_AA)
            
            self.fig = Image.fromarray(self.fig)
            img_draw = ImageDraw.Draw(self.fig)
            # text_bbox_size = img_draw.textsize(length, font=self.FT)
            text_bbox = img_draw.textbbox((0, 0), length, font=self.FT)
            text_bbox_size = (text_bbox[2] - text_bbox[0] + 20, 
                              text_bbox[3] - text_bbox[1] + 20)


            text_image = Image.new('RGBA', text_bbox_size, color=None)
            text_draw = ImageDraw.Draw(text_image)
            text_draw.text((10, 10), length, fill=self.c_color, font=self.FT)
            
            # text_draw.rectangle([(10, 10), (10+text_bbox[2] - text_bbox[0], 10+text_bbox[3] - text_bbox[1])], outline="black", width=2)
            # text_draw.rectangle([(2, 2), (text_image.width-2, text_image.height-2)], outline="black", width=2)

            # text_image = text_image.rotate(-angle, expand=False)
            text_pos = (int(xc - text_image.width / 2), 
                        int(yc - text_image.height / 2))
            self.fig.paste(text_image, text_pos, text_image)
            self.fig = np.asarray(self.fig)
            
            # cv2.circle(self.fig, text_pos, 4, color=self.p_color, thickness=-1, lineType=cv2.LINE_AA)
        
        
    def plot_measure_of_angle(self, measure_of_angle):
        for angle, measure in measure_of_angle:
            if '+' in measure or '-' in measure:
                measure = f"({measure})°"
            else:
                measure = f"{measure}°"
            xa, ya = self.p_pos[angle[0]]
            xb, yb = self.p_pos[angle[1]]
            xc, yc = self.p_pos[angle[2]]
            
            BA_length = self.distance([xb, yb], [xa, ya])
            BC_length = self.distance([xb, yb], [xc, yc])
            BA_unit = ((xa - xb) / BA_length, (ya - yb) / BA_length)
            BC_unit = ((xc - xb) / BC_length, (yc - yb) / BC_length)
            BD_vec = (BA_unit[0] + BC_unit[0], BA_unit[1] + BC_unit[1])
            
            # base position at angle bisector
            xd = xb + BD_vec[0] * 40
            yd = yb + BD_vec[1] * 40
            
            # cv2.circle(self.fig, (int(xd), int(yd)), 4, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            # draw arc for angle ABC
            line_len = min(self.distance([xa, ya], [xb, yb]),
                           self.distance([xc, yc], [xb, yb]))
            radius = min([20, int(line_len / 5)])
            # ensure BA -> BC clock wise, BA x BC < 0 
            # for cv2, ensure BA -> BC counter clock wise, BA x BC > 0 
            # BA = [xa - xb, ya - yb]
            # BC = [xc - xb, yc - yb]
            if (xa - xb) * (yc - yb) - (ya - yb) * (xc - xb) < 0:
                xa, ya, xc, yc = xc, yc, xa, ya
            angle_BC = np.degrees(np.arctan2(yc-yb, xc-xb))
            angle_BA = np.degrees(np.arctan2(ya-yb, xa-xb))
            angle_BC = angle_BC + 360 if angle_BC < 0 else angle_BC
            angle_BA = angle_BA + 360 if angle_BA < 0 else angle_BA
            if angle_BC < angle_BA:
                angle_BA = angle_BA - 360
            # print(angle_BA, angle_BC)
            # B as circle center
            cv2.ellipse(self.fig, (xb, yb), (radius, radius), 0, angle_BA, angle_BC, self.a_color, self.line_width, lineType=cv2.LINE_AA)
            
            self.fig = Image.fromarray(self.fig)
            img_draw = ImageDraw.Draw(self.fig)
            text_bbox = img_draw.textbbox((0, 0), measure, font=self.FT)
            text_bbox_size = (text_bbox[2] - text_bbox[0], 
                              text_bbox[3] - text_bbox[1])
            # text_bbox_size = img_draw.textsize(measure, font=self.FT)
            text_pos = [int(max(xd - text_bbox_size[0] / 2, 5)),  
                       int(max(yd - text_bbox_size[1] / 2, 5))]
            img_draw.text(text_pos, measure, font=self.FT, fill=self.c_color)
            self.fig = np.asarray(self.fig)
            
            
    
    def plot_annotation(self):
        self.plot_fill_in_color(self.annotation_targets['filled_entity'])
        self.plot_right_angle(self.annotation_targets['right_angles'])
        self.plot_equal_lines(self.annotation_targets['eq_lines'])
        self.plot_equal_anlges(self.annotation_targets['eq_angles'])
        self.plot_equal_arcs(self.annotation_targets['eq_arcs'])
        self.plot_length_of_line(self.annotation_targets['length_of_line'])
        self.plot_measure_of_angle(self.annotation_targets['measure_of_angle'])
        
    # def plot_value(self):
    #     self.plot_length_of_line(self.annotation_targets['length_of_line'])
    #     self.plot_measure_of_angle(self.annotation_targets['measure_of_angle'])
    
    def formulate_caption(self):
        caption_str = random.choice([
            "The diagram shows a geometric figure with ",
            "This is a geometric diagram featuring ",
            "The figure depicts a geometric diagram involving ",
            "In this diagram, the geometric pattern consists of ",
            "The geometric construction in this figure includes "
        ])
        # summarize of relation
        relations = []
        for clause in self.text_cdls:
            predicate, _ = parse_clause(clause)
            if predicate in PREDICATES_ENT:
                relations.append(convert_upper_to_lower(predicate))
            elif predicate in PREDICATES_REL or predicate in PREDICATES_REL_2:
                relations.append(PREDICATES_TO_NAMES[predicate])
        relations = list(set(relations))
        if len(relations) == 1:
            caption_str += f"{relations[0]}. "
        elif len(relations) == 2:
            caption_str += f"{relations[0]} and {relations[1]}. "
        else:
            caption_str += f"{', '.join(relations[:-1])} and {relations[-1]}. "
        caption_str += "Here is a detailed description:\n"
        # points
        points = list(self.p_pos.keys())
        caption_str += f"Points: $ {', '.join(points)} $ .\n"

        # lines or circle
        lines = [''.join(l) for l in self.lines]
        caption_str += f"Lines: $ {', '.join(lines)} $ .\n"
        if len(self.circles) != 0:
            caption_str += f"Circles: $ {', '.join(self.circles)} $ .\n"
        else:
            caption_str += f"Circles: there's no circle in the diagram.\n"
        # clause
        caption_str += f"Geometry Relations:\n- "
        caption_list = clause_to_nature_language(
            self.text_cdls + self.construct_cdls,
            self.natural_template
        )
        caption_str += '.\n- '.join(caption_list) + '. '
        if self.debug:
            print(caption_str)
        self.caption_str = caption_str
        
        
    def plot(self):
        # plot annotations first
        self.plot_annotation()
        
        # plot points
        for p, pos in self.p_pos.items():
            cv2.circle(self.fig, pos, 4, color=self.p_color, thickness=-1, lineType=cv2.LINE_AA)
        
        # plot lines
        for line in self.lines:
            for i in range(len(line)-1):
                p1_pos, p2_pos = self.p_pos[line[i]], self.p_pos[line[i+1]]
                cv2.line(self.fig, p1_pos, p2_pos, self.l_color, self.line_width, lineType=cv2.LINE_AA)
                
        # plot circles
        for circle in self.circles:
            c_pos = self.p_pos[circle]
            radius = self.get_radius(circle)
            cv2.circle(self.fig, c_pos, int(radius), color=self.l_color, 
                       thickness=self.line_width, lineType=cv2.LINE_AA)
        
        # plot chars
        for point_i in self.p_pos:
            # char, font, size, width
            text_width = 2
            text_size = 1
            text_bbox_size, _ = cv2.getTextSize(point_i, self.font, text_size, text_width)
            # text_pos is the top left (in image) position
            text_pos = self.get_best_chars_position(text_bbox_size, self.p_pos[point_i])
            cv2.putText(self.fig, point_i, text_pos, self.font, text_size, 
                        self.c_color, text_width, cv2.LINE_AA)
        
        self.formulate_caption()


    def save_fig(self, fig_name, fig_dir):
        if '.png' in fig_name or '.jpg' in fig_name:
            cv2.imwrite(f"{fig_dir}/{fig_name}", self.fig)
        else:
            cv2.imwrite(f"{fig_dir}/{fig_name}.png", self.fig)
            
if __name__ == '__main__':
    setup_seed(124)
    # dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
    predicate_GDL = json.load(open('json/predicate_GDL.json', 'r', encoding='utf-8'))
    theorem_GDL = json.load(open('json/theorem_GDL.json', 'r', encoding='utf-8'))
    for i in range(100):
        # clauses_base = random.choices(PREDICATES_ENT + PREDICATES_REL_2, k=1)
        clauses_base = random.choices(PREDICATES_ENT, k=1)
        clauses_rel = random.choices(PREDICATES_REL, k=2)
        # clauses_base = [
        #     "Trapezoid",
        # ]
        # clauses_rel = [
        #     'IsCircumcenterOfQuadrilateral', 
        #     'IsMidpointOfArc',
        #     # 'IsAltitudeOfQuadrilateral',
        #     # 'IsIncenterOfTriangle',
        #     # "IsAltitudeOfTriangle",
        #     # "IsCircumcenterOfQuadrilateral",
        #     # "IsMidpointOfArc"
        #     ]
        print('---------- Chosen Predicates ----------')
        print('clauses_base: ', clauses_base)
        print('clauses_rel: ', clauses_rel)
        
        cg = ClauseGenerator(predicate_GDL, theorem_GDL)
        cg.empty_states()
        c_cdls, t_cdls = cg.generate_clauses_from_predicates(
            clauses_base, 
            clauses_rel, 
            n_more_lines=0
        )
        states = cg.states
        
        # states = {'points': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 'lines': [('b', 'c'), ('e', 'f'), ('a', 'b', 'e'), ('c', 'd', 'f'), ('g', 'h'), ('c', 'd'), ('a', 'c', 'g'), ('a', 'd', 'h')], 'circles': [], 'polygons': [('a', 'b', 'c'), ('a', 'b', 'c', 'd'), ('a', 'c', 'd'), ('a', 'g', 'h')], 'constraints': ['Equal(MeasureOfAngle(cab),90)', 'Equal(LengthOfLine(ab),LengthOfLine(ac))', 'Equal(LengthOfLine(ae),LengthOfLine(eb))', 'Equal(LengthOfLine(cf),LengthOfLine(fd))', 'Polygon(abcd)', 'Collinear(aeb)', 'Collinear(dfc)', 'Equal(LengthOfLine(ag),LengthOfLine(gc))', 'Equal(LengthOfLine(ah),LengthOfLine(hd))', 'Collinear(agc)', 'Collinear(ahd)'], 'constraints_base': ['Equal(MeasureOfAngle(cab),90)', 'Equal(LengthOfLine(ab),LengthOfLine(ac))'], 'points_on_circle': {}}
        # c_cdls = ['Shape(ab,bc,ca)', 'Polygon(abcd)', 'Shape(ef)', 'Shape(ab,bc,cd,da)', 'Collinear(aeb)', 'Collinear(dfc)', 'Shape(gh)', 'Shape(ac,cd,da)', 'Collinear(agc)', 'Collinear(ahd)']
        # t_cdls = ['IsoscelesRightTriangle(abc)', 'IsMidsegmentOfQuadrilateral(ef,abcd)', 'IsMidsegmentOfTriangle(gh,acd)']
        
        print('---------- Allocator Inputs ----------')
        print(states)
        print('c_cdls: ', c_cdls)
        print('t_cdls: ', t_cdls)

        allocator = Allocator(states, c_cdls, t_cdls, allocate_value=True)
        print('---------- Formulated CDLs ----------')
        
            
        allocator.allocate()
        print('Text CDLs: ')
        for t_cdl in allocator.formulated_cdls['text_cdls']:
            print('\t', t_cdl)
        print('Construct CDLs: ')
        for c_cdl in allocator.formulated_cdls['construct_cdls']:
            print('\t', c_cdl)
            
        plotter = Plotter(allocator.states, 
                          allocator.formulated_cdls['text_cdls'],
                          allocator.formulated_cdls['construct_cdls'],
                          allocator.formulated_cdls['image_cdls'],
                          replace_characters=True,
                          debug=True)
        print('---------- Annotation Targets ----------')
        print(plotter.annotation_targets)
        
        plotter.plot()
        plotter.save_fig('test', 'imgs_test')
        
        
        print('==============================================')
