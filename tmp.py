
def bgr_to_hex(b, g, r):
    # 将每个分量转换为两位的十六进制字符串
    rb = format(r, '02x')
    gb = format(g, '02x')
    bb = format(b, '02x')
    
    # 拼接成完整的十六进制颜色码
    hex_color = f"#{rb}{gb}{bb}"
    return hex_color


colors = [
        (151, 85, 47),
        (204, 72, 6),
        (21, 80, 240),
        (208, 211, 208),
        (150, 88, 0),
        (151, 85, 47),
        (42, 42, 42),
        (201, 176, 200),
        (208, 211, 208)
    ]


for c in colors:
    hex_color = bgr_to_hex(*c)
    print(f'bgr: {c}, hex: {hex_color}')
