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
'solver' invokes other modules to enable interactive problem-solving and automated problem-solving.
The automated problem-solving implements both forward search and backward search, allowing for the
configuration of various search strategies (breadth-first, depth-first, random, beam).
"""

__all__ = [
    "Interactor", "ForwardSearcher", "BackwardSearcher"
]

from formalgeo_v2.solver.backward_search import BackwardSearcher
from formalgeo_v2.solver.forward_search import ForwardSearcher
from formalgeo_v2.solver.interactive import Interactor
