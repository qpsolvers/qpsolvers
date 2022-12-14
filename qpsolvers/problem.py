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
Model for a quadratic program.
"""

from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .check_problem_constraints import check_problem_constraints
from .conversions import linear_from_box_inequalities
from .solve_qp import solve_qp


class Problem:

    """
    Model for a quadratic program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\mbox{minimize} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    This is a convenience class providing

    Attributes
    ----------
    P :
        Symmetric quadratic-cost matrix (most solvers require it to be definite
        as well).
    q :
        Quadratic-cost vector.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    A :
        Linear equality matrix.
    b :
        Linear equality vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    """

    P: Union[np.ndarray, spa.csc_matrix]
    q: np.ndarray
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None
    h: Optional[np.ndarray] = None
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None
    b: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None

    def __init__(
        self,
        P: Union[np.ndarray, spa.csc_matrix],
        q: np.ndarray,
        G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        b: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        self.P = P
        self.q = q
        self.G = G
        self.h = h
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub

    def check_constraints(self):
        """
        Check that problem constraints are properly specified.
        """
        return check_problem_constraints(self.G, self.h, self.A, self.b)

    def solve(self, solver: str, **kwargs):
        """
        Solve problem with a given QP solver.

        Parameters
        ----------
        solver :
            Name of the QP solver.
        """
        return solve_qp(
            self.P,
            self.q,
            self.G,
            self.h,
            self.A,
            self.b,
            self.lb,
            self.ub,
            solver=solver,
            **kwargs
        )
