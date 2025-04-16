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

"""Download and Management of Datasets and Formal Systems."""

__all__ = [
    "show_available_datasets", "download_dataset", "remove_dataset",
    "DatasetLoader"
]

from formalgeo_v2.data.data import show_available_datasets, download_dataset, remove_dataset
from formalgeo_v2.data.data import DatasetLoader
