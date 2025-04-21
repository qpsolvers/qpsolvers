#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `SCS <https://www.cvxgrp.org/scs/>`__.

SCS (Splitting Conic Solver) is a numerical optimization package for solving
large-scale convex quadratic cone problems, which is a general class of
problems that includes quadratic programming. If you use SCS in a scientific
work, consider citing the corresponding paper [ODonoghue2021]_.
"""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as spa
from numpy import ndarray
from scipy.sparse import csc_matrix
from scs import solve

from ..conversions import ensure_sparse_matrices
from ..problem import Problem
from ..solution import Solution
from ..solve_unconstrained import solve_unconstrained

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
    """Add box cone to the problem.

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


def scs_solve_problem(
    problem: Problem,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using SCS.

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
    P, G, A = ensure_sparse_matrices("scs", P, G, A)
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
        warnings.warn(
            "QP is unconstrained: solving with SciPy's LSQR rather than SCS"
        )
        return solve_unconstrained(problem)
    if lb is not None or ub is not None:
        __add_box_cone(n, lb, ub, cone, data)
    kwargs["verbose"] = verbose
    result = solve(data, cone, **kwargs)

    solution = Solution(problem)
    solution.extras = result["info"]
    status_val = result["info"]["status_val"]
    solution.found = status_val == 1
    if not solution.found:
        warnings.warn(
            f"SCS returned {status_val}: {__status_val_meaning__[status_val]}"
        )
    solution.x = result["x"]
    meq = A.shape[0] if A is not None else 0
    solution.y = result["y"][:meq] if A is not None else np.empty((0,))
    solution.z = (
        result["y"][meq : meq + G.shape[0]]
        if G is not None
        else np.empty((0,))
    )
    solution.z_box = (
        -result["y"][-n:]
        if lb is not None or ub is not None
        else np.empty((0,))
    )
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
    r"""Solve a quadratic program using SCS.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h                \\
                & A x = b                   \\
                & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `SCS <https://github.com/cvxgrp/scs>`__.

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
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = scs_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
