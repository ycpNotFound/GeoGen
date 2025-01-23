PREDICATES_PRE = [
    # Preset
    "Collinear",
    "Cocircular",
]
PREDICATES_ENT = [
    # Entity
    # "Circle",
    "Triangle",
    "Parallelogram",
    "Rectangle",
    "Rhombus",
    "RightTriangle",
    "Square",
    "EquilateralTriangle",
    "IsoscelesTriangle",
    "Trapezoid",
    "Kite",
    "RightTrapezoid",
    "IsoscelesTrapezoid",
    "IsoscelesRightTriangle",
]
PREDICATES_REL = [
    # Relation
    # "IsCentreOfCircle", 
    "PerpendicularBetweenLine",
    "ParallelBetweenLine",
    "IsTangentOfCircle",
    "IsDiameterOfCircle",
    "IsMidpointOfLine",
    "IsBisectorOfAngle",
    "IsPerpendicularBisectorOfLine",
    # "SimilarBetweenTriangle",
    # "SimilarBetweenQuadrilateral",
    "IsAltitudeOfTriangle",
    # "MirrorCongruentBetweenTriangle",
    # "CongruentBetweenTriangle",
    "IsMedianOfTriangle",
    "IsIncenterOfTriangle",
    "IsMidsegmentOfTriangle",
    # "MirrorSimilarBetweenTriangle",
    "IsMidpointOfArc",
    # "CongruentBetweenArc",
    "IsMidsegmentOfQuadrilateral",
    "IsCentroidOfTriangle",
    # "MirrorCongruentBetweenQuadrilateral",
    # "IsAltitudeOfQuadrilateral",
    # "CongruentBetweenQuadrilateral",
    "IsCircumcenterOfTriangle",
    # "IsIncenterOfQuadrilateral",
    "IsCircumcenterOfQuadrilateral",
]
PREDICATES_REL_2 = [
    "SimilarBetweenTriangle",
    # "MirrorSimilarBetweenTriangle",
    "SimilarBetweenQuadrilateral",
    "CongruentBetweenTriangle",
    # "MirrorCongruentBetweenTriangle",
    # "CongruentBetweenArc",
    "CongruentBetweenQuadrilateral",
    # "MirrorCongruentBetweenQuadrilateral",
]

PREDICATES_REL_3 = [
    "IsCentreOfCircle", 
    "CongruentBetweenArc",
    "IsAltitudeOfQuadrilateral",
    "IsIncenterOfQuadrilateral",
    "SimilarBetweenArc",
    "IsOrthocenterOfTriangle",
]

PREDICATES_ATTR = [
    # Attribution
    "LengthOfLine",
    "MeasureOfAngle",
    "MeasureOfArc",
    "RadiusOfCircle",
    "AreaOfTriangle",
    "DiameterOfCircle",
    "AreaOfQuadrilateral",
    "PerimeterOfTriangle",
    "LengthOfArc",
    "PerimeterOfQuadrilateral",
    "RatioOfSimilarTriangle",
    "HeightOfQuadrilateral",
    "AreaOfSector",
    "AreaOfCircle",
    "RatioOfSimilarArc",
    "RatioOfMirrorSimilarTriangle",
    "RatioOfSimilarQuadrilateral",
]


PREDICATES = PREDICATES_PRE + PREDICATES_ENT + PREDICATES_ATTR + PREDICATES_REL + PREDICATES_REL_2

PREDICATES_TO_NAMES = {
    "PerpendicularBetweenLine": "perpendicular lines",
    "ParallelBetweenLine": "parallel lines",
    "IsTangentOfCircle": "tangent line",
    "IsDiameterOfCircle": "diameter of circle",
    "IsMidpointOfLine": "midpoint",
    "IsBisectorOfAngle": "angle bisector",
    "IsPerpendicularBisectorOfLine": "perpendicular bisector",
    "IsAltitudeOfTriangle": "altitude line",
    "IsMedianOfTriangle": "median line",
    "IsIncenterOfTriangle": "incenter",
    "IsMidsegmentOfTriangle": "midegment",
    "IsMidpointOfArc": "midpoint",
    "IsMidsegmentOfQuadrilateral": "midsegment",
    "IsCentroidOfTriangle": "centroid",
    "IsCircumcenterOfTriangle": "circumcenter",
    "IsCircumcenterOfQuadrilateral": "circumcenter",
    "SimilarBetweenTriangle": "similar triangles",
    "SimilarBetweenQuadrilateral": "similar quadrilaterals",
    "CongruentBetweenTriangle": "congruent triangles",
    "CongruentBetweenQuadrilateral": "congruent quadrilaterals"
}
PRESET_COLOR_PROBS = [0.5, 0.06, 0.06, 0.06, 0.06, 0.06, 0.2]
PRESET_COLORS = [
    # l: line | p: point | c: char | a: annoation | f: fill_in
    # Color: BGR
    {
        "l_color": "#000000",
        "p_color": "#2f5597",
        "c_color": "#0648cc",
        "a_color": "#f05015",
        "f_color": "#ffffff"   
    },
    {
        "l_color": "#2a2a2a",
        "p_color": "#000000",
        "c_color": "#2a2a2a",
        "a_color": "#2f5597",
        "f_color": "#D1D3D0"
        
    },
    {
        "l_color": "#005896",
        "p_color": "#00406C",
        "c_color": "#2a2a2a",
        "a_color": "#f05015",
        "f_color": "#9DC7DD"
    },
    {
        "l_color": "#638E50",
        "p_color": "#48683A",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#B5D69E"
    },
    {
        "l_color": "#D29A42",
        "p_color": "#956A23",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#FFE6B2"
    },
    {
        "l_color": "#6B4582",
        "p_color": "#442C52",
        "c_color": "#2a2a2a",
        "a_color": "#DA3585",
        "f_color": "#C8B0C9"
    },
    {
        "l_color": "#282828",
        "p_color": "#000000",
        "c_color": "#282828",
        "a_color": "#282828",
        "f_color": "#ffffff"
    },
]

def hex_to_bgr(hex_str):
    rgb_color = tuple(int(hex_str[i:i+2], 16) for i in (1, 3, 5)) 
    bgr_color = rgb_color[::-1]  # Convert to BGR format 
    return bgr_color

def bgr_to_hex(b, g, r):
    # 将每个分量转换为两位的十六进制字符串
    rb = format(r, '02x')
    gb = format(g, '02x')
    bb = format(b, '02x')
    
    # 拼接成完整的十六进制颜色码
    hex_color = f"#{rb}{gb}{bb}"
    return hex_color


SYMBOL_MAPPING_1 = {
    "\\triangle": "triangle",
    "\\perp": "is perpendicular to",
    "\\parallel": "is parallel to",
    "\\odot": "circle",
    "\\angle": "angle",
    "\\arc": "arc",
    "\\sqrt": "sqrt of",
}
SYMBOL_MAPPING_2 = {
    "\\triangle": "△",
    "\\perp": "⊥",
    "\\parallel": "∥",
    "\\odot": "⊙",
    "\\angle": "∠",
    "\\arc": "⌒",
    "\\sqrt": "√"
}