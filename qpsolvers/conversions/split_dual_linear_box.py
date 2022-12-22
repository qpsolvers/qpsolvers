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
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Separate linear and box multipliers from a stacked vector of
    inequality-constraint dual variables.

    This function assumes linear and box inequalities were combined using
    :func:`qpsolvers.conversions.linear_from_box_inequalities`.

    Parameters
    ----------
    z_stacked :
        Stacked vector of dual multipliers.
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
        n_lb = lb.shape[0]
        n_ub = ub.shape[0]
        n_box = n_lb + n_ub
        z_box = z_stacked[-n_ub:] - z_stacked[-n_box:-n_ub]
        z = z_stacked[:-n_box]
    elif ub is not None:  # lb is None
        n_ub = ub.shape[0]
        z_box = z_stacked[-n_ub:]
        z = z_stacked[:-n_ub]
    elif lb is not None:  # ub is None
        n_lb = lb.shape[0]
        z_box = -z_stacked[-n_lb:]
        z = z_stacked[:-n_lb]
    else:  # lb is None and ub is None
        z = z_stacked
    return z, z_box
