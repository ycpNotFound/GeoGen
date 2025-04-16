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
'Problem' preserves all details of the problem-solving process, ensures the correctness and
consistency of the problem input conditions, and implements automatic diagram construction,
condition auto-expansion, and validity checks.
"""

__all__ = [
    "Problem"
]

from formalgeo_v2.problem.problem import Problem
