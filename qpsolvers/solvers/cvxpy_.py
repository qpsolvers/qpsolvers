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

"""Solver interface for CVXPY"""

from typing import Optional

import numpy as np
from cvxpy import Constant, Minimize, Problem, Variable, quad_form
from numpy import array

from .conversions import linear_from_box_inequalities


def cvxpy_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b                    \\\\
            & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    calling a given solver using the `CVXPY <http://www.cvxpy.org/>`_ modelling
    language.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector (not used).
    solver :
        Solver name in ``cvxpy.installed_solvers()``.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        print("CVXPY: note that warm-start values are ignored by wrapper")
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    n = q.shape[0]
    x = Variable(n)
    P_cst = Constant(P)  # see http://www.cvxpy.org/en/latest/faq/
    objective = Minimize(0.5 * quad_form(x, P_cst) + q @ x)
    constraints = []
    if G is not None:
        constraints.append(G @ x <= h)
    if A is not None:
        constraints.append(A @ x == b)
    prob = Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)
    if x.value is None:
        return None
    return array(x.value).reshape((n,))
