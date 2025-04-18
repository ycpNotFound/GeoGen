# Modified from FormalGeo (https://github.com/FormalGeo/FormalGeo)
# Original Copyright (C) 2022-2024 FormalGeo Development Team
# Author: Xiaokai Zhang
# Contact: formalgeo@gmail.com
#
# Modifications Copyright (C) 2025 Yicheng Pan
# Contact: yichpan@mail.ustc.edu.cn
#
# This file is licensed under the MIT License.
# See the LICENSE file in the root directory for details.

"""
'tools' provides some practical tools, such as problem-solving process outputting.
"""

__all__ = [
    "load_json", "save_json", "safe_save_json", "debug_print", "rough_equal", "get_user_input",
    "simple_show", "show_solution", "get_used_pid_and_theorem",
    "get_meta_hypertree",
    "get_solution_hypertree", "draw_solution_hypertree", "get_theorem_dag", "draw_theorem_dag"
]

from formalgeo_v2.tools.utils import load_json, save_json, safe_save_json, debug_print, rough_equal, get_user_input
from formalgeo_v2.tools.output import simple_show, show_solution, get_used_pid_and_theorem
from formalgeo_v2.tools.output import get_meta_hypertree
from formalgeo_v2.tools.output import get_solution_hypertree, draw_solution_hypertree, get_theorem_dag, draw_theorem_dag
