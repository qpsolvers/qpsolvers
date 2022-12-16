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
Solver interface for `SCS <https://www.cvxgrp.org/scs/>`__.

SCS (Splitting Conic Solver) is a numerical optimization package for solving
large-scale convex quadratic cone problems, which is a general class of
problems that includes quadratic programming. If you use SCS in some academic
work, consider citing the corresponding paper [ODonoghue2021]_.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as spa
from numpy import ndarray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scs import solve

from ..conversions import warn_about_sparse_conversion
from ..problem import Problem
from ..solution import Solution

# See https://www.cvxgrp.org/scs/api/exit_flags.html#exit-flags
__status_val_meaning__ = {
    -7: "INFEASIBLE_INACCURATE",
    -6: "UNBOUNDED_INACCURATE",
    -5: "SIGINT",
    -4: "FAILED",
    -3: "INDETERMINATE",
    -2: "INFEASIBLE (primal infeasible, dual unbounded)",
    -1: "UNBOUNDED (primal unbounded, dual infeasible)",
    0: "UNFINISHED (never returned, used as placeholder)",
    1: "SOLVED",
    2: "SOLVED_INACCURATE",
}


def __add_box_cone(
    n: int,
    lb: Optional[ndarray],
    ub: Optional[ndarray],
    cone: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """
    Add box cone to the problem.

    Parameters
    ----------
    n :
        Number of optimization variables.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    cone :
        SCS cone dictionary.
    data :
        SCS data dictionary.

    Notes
    -----
    See the `SCS Cones <https://www.cvxgrp.org/scs/api/cones.html>`__
    documentation for details.
    """
    cone["bl"] = lb if lb is not None else np.full((n,), -np.inf)
    cone["bu"] = ub if ub is not None else np.full((n,), +np.inf)
    zero_row = csc_matrix((1, n))
    data["A"] = spa.vstack(
        ((data["A"],) if "A" in data else ()) + (zero_row, -spa.eye(n)),
        format="csc",
    )
    data["b"] = np.hstack(
        ((data["b"],) if "b" in data else ()) + (1.0, np.zeros(n))
    )


def __solve_unconstrained(problem: Problem) -> Solution:
    """
    Solve an unconstrained quadratic program, warning if it is unbounded.

    Parameters
    ----------
    problem :
        Unconstrained quadratic program.

    Returns
    -------
    :
        Solution to the unconstrained QP, if it is bounded.

    Raises
    ------
    ValueError
        If the quadratic program is not unbounded below.
    """
    P, q, _, _, _, _, _, _ = problem.unpack()
    solution = Solution(problem)
    solution.x = lsqr(P, -q)[0]
    cost_check = np.linalg.norm(P @ solution.x + q)
    if cost_check > 1e-8:
        raise ValueError(
            f"problem is unbounded below (cost_check={cost_check:.1e}), "
            "q has component in the nullspace of P"
        )
    return solution


def __ensure_sparse_matrices(
    P: Union[ndarray, csc_matrix],
    G: Optional[Union[ndarray, csc_matrix]],
    A: Optional[Union[ndarray, csc_matrix]],
) -> Tuple[csc_matrix, Optional[csc_matrix], Optional[csc_matrix]]:
    """
    Make sure matrices are sparse.
    """
    if isinstance(P, ndarray):
        warn_about_sparse_conversion("P")
        P = csc_matrix(P)
    if isinstance(G, ndarray):
        warn_about_sparse_conversion("G")
        G = csc_matrix(G)
    if isinstance(A, ndarray):
        warn_about_sparse_conversion("A")
        A = csc_matrix(A)
    return P, G, A


def scs_solve_problem(
    problem: Problem,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """
    Solve a quadratic program using `SCS <https://github.com/cvxgrp/scs>`__.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Raises
    ------
    ValueError
        If the quadratic program is not unbounded below.

    Notes
    -----
    Keyword arguments are forwarded as is to SCS. For instance, we can call
    ``scs_solve_qp(P, q, G, h, eps_abs=1e-6, eps_rel=1e-4)``. SCS settings
    include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iters``
         - Maximum number of iterations to run.
       * - ``time_limit_secs``
         - Time limit for solve run in seconds (can be fractional). 0 is
           interpreted as no limit.
       * - ``eps_abs``
         - Absolute feasibility tolerance. See `Termination criteria
           <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`__.
       * - ``eps_rel``
         - Relative feasibility tolerance. See `Termination criteria
           <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`__.
       * - ``eps_infeas``
         - Infeasibility tolerance (primal and dual), see `Certificate of
           infeasibility
           <https://www.cvxgrp.org/scs/algorithm/index.html#certificate-of-infeasibility>`_.
       * - ``normalize``
         - Whether to perform heuristic data rescaling. See `Data equilibration
           <https://www.cvxgrp.org/scs/algorithm/equilibration.html#equilibration>`__.

    Check out the `SCS settings
    <https://www.cvxgrp.org/scs/api/settings.html#settings>`_ documentation for
    all available settings.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = __ensure_sparse_matrices(P, G, A)
    n = P.shape[0]

    data: Dict[str, Any] = {"P": P, "c": q}
    cone: Dict[str, Any] = {}
    if initvals is not None:
        data["x"] = initvals
    if A is not None and b is not None:
        if G is not None and h is not None:
            data["A"] = spa.vstack([A, G], format="csc")
            data["b"] = np.hstack([b, h])
            cone["z"] = b.shape[0]  # zero cone
            cone["l"] = h.shape[0]  # positive cone
        else:  # G is None and h is None
            data["A"] = A
            data["b"] = b
            cone["z"] = b.shape[0]  # zero cone
    elif G is not None and h is not None:
        data["A"] = G
        data["b"] = h
        cone["l"] = h.shape[0]  # positive cone
    elif lb is None and ub is None:  # no constraint
        return __solve_unconstrained(problem)
    if lb is not None or ub is not None:
        __add_box_cone(n, lb, ub, cone, data)
    kwargs["verbose"] = verbose
    result = solve(data, cone, **kwargs)

    solution = Solution(problem)
    solution.extras = result["info"]
    status_val = result["info"]["status_val"]
    if status_val != 1:
        warnings.warn(
            f"SCS returned {status_val}: {__status_val_meaning__[status_val]}"
        )
        return solution
    solution.x = result["x"]
    meq = A.shape[0] if A is not None else 0
    if A is not None:
        solution.y = result["y"][:meq]
    if G is not None:
        solution.z = result["y"][meq : meq + G.shape[0]]
    if lb is not None or ub is not None:
        solution.z_box = -result["y"][-n:]
    return solution


def scs_solve_qp(
    P: Union[ndarray, csc_matrix],
    q: ndarray,
    G: Optional[Union[ndarray, csc_matrix]] = None,
    h: Optional[ndarray] = None,
    A: Optional[Union[ndarray, csc_matrix]] = None,
    b: Optional[ndarray] = None,
    lb: Optional[ndarray] = None,
    ub: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `SCS <https://github.com/cvxgrp/scs>`__.

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
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ValueError
        If the quadratic program is not unbounded below.

    Notes
    -----
    Keyword arguments are forwarded as is to SCS. For instance, we can call
    ``scs_solve_qp(P, q, G, h, eps_abs=1e-6, eps_rel=1e-4)``. SCS settings
    include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iters``
         - Maximum number of iterations to run.
       * - ``time_limit_secs``
         - Time limit for solve run in seconds (can be fractional). 0 is
           interpreted as no limit.
       * - ``eps_abs``
         - Absolute feasibility tolerance. See `Termination criteria
           <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`__.
       * - ``eps_rel``
         - Relative feasibility tolerance. See `Termination criteria
           <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`__.
       * - ``eps_infeas``
         - Infeasibility tolerance (primal and dual), see `Certificate of
           infeasibility
           <https://www.cvxgrp.org/scs/algorithm/index.html#certificate-of-infeasibility>`_.
       * - ``normalize``
         - Whether to perform heuristic data rescaling. See `Data equilibration
           <https://www.cvxgrp.org/scs/algorithm/equilibration.html#equilibration>`__.

    Check out the `SCS settings
    <https://www.cvxgrp.org/scs/api/settings.html#settings>`_ documentation for
    all available settings.
    """
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = scs_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x
