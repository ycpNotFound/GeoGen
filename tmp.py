from sympy import *

x, y, z = symbols('x y z')
eq1 = Eq(x + 3*y - z, 1)
eq2 = Eq(x - y + 2*z, -2)
eq3 = Eq(3*x - 2*y + z, 0)

solution = solve((eq1, eq2, eq3), (x, y, z))  # 解方程
print("解为:", solution)

# 显示逐步求解
steps = solve((eq1, eq2, eq3), (x, y, z))
for step in steps:
    print(step)