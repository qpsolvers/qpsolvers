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
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .exceptions import NoSolverSelected
from .problem import Problem
from .solve_problem import solve_problem
from .solvers import available_solvers


def solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
    initvals: Optional[np.ndarray] = None,
    sym_proj: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Solve a quadratic program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using the QP solver selected by the ``solver`` keyword argument.

    Parameters
    ----------
    P :
        Symmetric cost matrix (most solvers require it to be definite as well).
    q :
        Cost vector.
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
    solver :
        Name of the QP solver, to choose in
        :data:`qpsolvers.available_solvers`. This argument is mandatory.
    initvals :
        Primal candidate vector :math:`x` values used to warm-start the solver.
    sym_proj :
        Set to ``True`` to project the cost matrix :math:`P` to its symmetric
        part. Some solvers assume :math:`P` is symmetric and will return
        unintended results if it is not the case.
    verbose :
        Set to ``True`` to print out extra information.

    Note
    ----
    In quadratic programming, the matrix :math:`P` should be symmetric. Many
    solvers (including CVXOPT, OSQP and quadprog) leverage this property and
    may return unintended results when it is not the case. You can set
    ``sym_proj=True`` to project :math:`P` on its symmetric part, at the cost
    of a little computation time.

    Returns
    -------
    :
        Optimal solution if found, otherwise ``None``.

    Raises
    ------
    NoSolverSelected
        If the ``solver`` keyword argument is not set.

    SolverNotFound
        If the requested solver is not in :data:`qpsolvers.available_solvers`.

    ValueError
        If the problem is not correctly defined. For instance, if the solver
        requires a definite cost matrix but the provided matrix :math:`P` is
        not.

    Notes
    -----
    Extra keyword arguments given to this function are forwarded to the
    underlying solver. For example, we can call OSQP with a custom absolute
    feasibility tolerance by ``solve_qp(P, q, G, h, solver='osqp',
    eps_abs=1e-6)``. See the :ref:`Supported solvers <Supported solvers>` page
    for details on the parameters available to each solver.

    There is no guarantee that a ``ValueError`` is raised if the provided
    problem is non-convex, as some solvers don't check for this. Rather, if the
    problem is non-convex and the solver fails because of that, then a
    ``ValueError`` will be raised.
    """
    if solver is None:
        raise NoSolverSelected(
            "Set the `solver` keyword argument to one of the "
            f"available solvers in {available_solvers}"
        )
    if sym_proj:
        P = 0.5 * (P + P.transpose())
        warnings.warn(
            "The `sym_proj` feature is deprecated "
            "and will be removed in qpsolvers v2.9",
            DeprecationWarning,
            stacklevel=2,
        )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = solve_problem(problem, solver, initvals, verbose, **kwargs)
    return solution.x
