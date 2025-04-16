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
'core' responsible for GPL statements executing, which consists of 2 submodules.
'GeometryPredicateLogicExecutor' responsible for GPL statements parsing and relational inference.
'EquationKiller' responsible for symbolic and algebraic computation.
"""

__all__ = [
    "GeometryPredicateLogicExecutor", "EquationKiller"
]

from formalgeo_v2.core.engine import (EquationKiller, GeometryPredicateLogicExecutor)
from formalgeo_v2.core.engine_v2 import (EquationKillerV2, GeometryPredicateLogicExecutorV2)
