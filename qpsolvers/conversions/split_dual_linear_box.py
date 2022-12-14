#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

"""
Convert stacked dual multipliers into linear and box multipliers.
"""

from typing import Optional, Tuple

import numpy as np


def split_dual_linear_box(
    z_stacked: np.ndarray,
    n: int,
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Separate linear and box multipliers from a stacked vector of dual
    variables.

    Parameters
    ----------
    z_stacked :
        Stacked vector of dual multipliers.
    n :
        Number of optimization variables.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.

    Returns
    -------
    :
        Pair :code:`z, z_box` of linear and box multipliers. Both can be
        `None` if there is no corresponding constraint.
    """
    z, z_box = None, None
    if lb is not None and ub is not None:
        z_box = z_stacked[-n:] - z_stacked[-2 * n : -n]
        z = z_stacked[: -2 * n]
    elif ub is not None:  # lb is None
        z_box = z_stacked[-n:]
        z = z_stacked[:-n]
    elif lb is not None:  # ub is None
        z_box = -z_stacked[-n:]
        z = z_stacked[:-n]
    else:  # lb is None and ub is None
        z = z_stacked
    return z, z_box
