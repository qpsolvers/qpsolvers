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
Output from a QP solver.
"""

from typing import Optional

import numpy as np


class Solution:

    """
    Output from a QP solver.

    Attributes
    ----------
    x :
        Primal solution (None if no solution was found).
    y :
        Dual multipliers for equality constraints (None if no solution was
        found).
    z :
        Dual multipliers for linear inequality constraints (None if no solution
        was found).
    z_box :
        Dual multipliers for box inequality constraints (None if no solution
        was found).
    extra :
        Other outputs, specific to each solver.
    """

    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    z: Optional[np.ndarray]
    z_box: Optional[np.ndarray]
    extras: dict

    @property
    def is_empty(self) -> bool:
        """
        True if the solution is empty.
        """
        return self.x is None
