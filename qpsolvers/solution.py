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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .problem import Problem


@dataclass(frozen=False)
class Solution:

    """
    Output from a QP solver.

    Attributes
    ----------
    extras :
        Other outputs, specific to each solver.
    obj :
        Primal objective at the solution (None if no solution was found).
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
    """

    problem: Problem
    extras: dict = field(default_factory=dict)
    obj: Optional[float] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    z_box: Optional[np.ndarray] = None

    @property
    def is_empty(self) -> bool:
        """
        True if the solution is empty.
        """
        return self.x is None

    def primal_residual() -> float:
        """
        Compute the primal residual of the solution.

        Notes
        -----
        See for instance [tolerances]_ for an overview of optimality conditions
        and why this residual will be zero at the optimum.
        """
        raise NotImplementedError()
