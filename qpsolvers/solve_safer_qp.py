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
Solve quadratic programs.
"""

import warnings
from typing import Optional

import numpy as np

from .exceptions import NoSolverSelected
from .solve_qp import solve_qp
from .solvers import dense_solvers


def solve_safer_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    sr: float,
    reg: float = 1e-8,
    solver: Optional[str] = None,
    initvals: Optional[np.ndarray] = None,
    sym_proj: bool = False,
) -> Optional[np.ndarray]:
    """
    Solve the "safer" Quadratic Program with repulsive inequality constraints,
    defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x +
                \\frac{1}{2} \\mathit{reg} \\|s\\|^2 - \\mathit{sr} \\1^T s
                \\\\
            \\mbox{subject to}
                & G x \\leq h
        \\end{array}\\end{split}

    Slack variables `s` (i.e. distance to inequality constraints) are added to
    the vector of optimization variables and included in the cost function.
    This pushes the solution of this "safer" QP is further inside the
    linear constraint region.

    Parameters
    ----------
    P :
        Symmetric cost matrix.
    q :
        Cost vector.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    sr :
        This is the "slack repulsion" parameter that makes inequality
        constraints repulsive. In practice it weighs the linear term on slack
        variables in the augmented cost function. Higher values bring the
        solution further inside the constraint region but override the
        minimization of the original objective.
    reg :
        Regularization term that weighs squared slack variables in the cost
        function. Increase this parameter in case of numerical instability, and
        otherwise set it as small as possible compared, so that the squared
        slack cost is as small as possible compared to the regular cost.
    solver :
        Name of the QP solver to use.
    initvals :
        Primal candidate vector `x` values used to warm-start the solver.
    sym_proj :
        Set to `True` when the `P` matrix provided is not symmetric.

    Returns
    -------
    :
        Optimal solution to the relaxed QP, if found, otherwise ``None``.

    Raises
    ------
    ValueError
        If the quadratic program is not feasible.

    Note
    ----
    This is a legacy function.

    Notes
    -----
    This method can be found in the Inverse Kinematics resolution of Nozawa et
    al. (Humanoids 2016). It also appears in earlier works such as the
    "optimally safe" tension distribution algorithm of Borgstrom et al. (IEEE
    Transactions on Robotics, 2009).
    """
    warnings.warn(
        "The `solve_safer_qp` function is deprecated "
        "and will be removed in qpsolvers v2.7",
        DeprecationWarning,
        stacklevel=2,
    )
    if solver is None:
        raise NoSolverSelected(
            "Set the `solver` keyword argument to one of the "
            f"available dense solvers in {dense_solvers}"
        )
    if solver not in dense_solvers:
        raise NotImplementedError(
            "This function is only available for dense solvers"
        )
    n, m = P.shape[0], G.shape[0]
    E, Z = np.eye(m), np.zeros((m, n))
    P2 = np.vstack([np.hstack([P, Z.T]), np.hstack([Z, reg * np.eye(m)])])
    q2 = np.hstack([q, -sr * np.ones(m)])
    G2 = np.hstack([Z, E])
    h2 = np.zeros(m)
    A2 = np.hstack([G, -E])
    b2 = h
    x = solve_qp(
        P2,
        q2,
        G2,
        h2,
        A2,
        b2,
        solver=solver,
        initvals=initvals,
        sym_proj=sym_proj,
    )
    if x is None:
        return None
    return x[:n]
