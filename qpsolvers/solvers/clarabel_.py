#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023 Inria

"""Solver interface for `Clarabel.rs`_.

.. _Clarabel.rs: https://github.com/oxfordcontrol/Clarabel.rs

Clarabel.rs is a Rust implementation of an interior point numerical solver for
convex optimization problems using a novel homogeneous embedding. A paper
describing the Clarabel solver algorithm and implementation will be forthcoming
soon (retrieved: 2023-02-06). Until then, the authors ask that you cite its
documentation if you have found Clarabel.rs useful in your work.
"""

import warnings
from typing import Optional, Union

import clarabel
import numpy as np
import scipy.sparse as spa

from ..conversions import (
    ensure_sparse_matrices,
    linear_from_box_inequalities,
    split_dual_linear_box,
)
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution
from ..solve_unconstrained import solve_unconstrained


def clarabel_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    r"""Solve a quadratic program using Clarabel.rs.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded as options to Clarabel.rs. For instance, we
    can call ``clarabel_solve_qp(P, q, G, h, u, tol_feas=1e-6)``. Clarabel
    options include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iter``
         - Maximum number of iterations.
       * - ``time_limit``
         - Time limit for solve run in seconds (can be fractional).
       * - ``tol_gap_abs``
         - absolute duality-gap tolerance
       * - ``tol_gap_rel``
         - relative duality-gap tolerance
       * - ``tol_feas``
         - feasibility check tolerance (primal and dual)

    Check out the `API reference
    <https://oxfordcontrol.github.io/ClarabelDocs/stable/api_settings/#Clarabel.Settings>`_
    for details.

    Lower values for absolute or relative tolerances yield more precise
    solutions at the cost of computation time. See *e.g.* [Caron2022]_ for a
    primer on solver tolerances and residuals.
    """
    if initvals is not None and verbose:
        warnings.warn("Clarabel: warm-start values are ignored")
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices("clarabel", P, G, A)
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=True)

    cones = []
    A_list = []
    b_list = []
    if A is not None and b is not None:
        A_list.append(A)
        b_list.append(b)
        cones.append(clarabel.ZeroConeT(b.shape[0]))
    if G is not None and h is not None:
        A_list.append(G)
        b_list.append(h)
        cones.append(clarabel.NonnegativeConeT(h.shape[0]))
    if not A_list:
        return solve_unconstrained(problem)

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    for key, value in kwargs.items():
        setattr(settings, key, value)

    A_stack = spa.vstack(A_list, format="csc")
    b_stack = np.concatenate(b_list)
    try:
        solver = clarabel.DefaultSolver(
            P, q, A_stack, b_stack, cones, settings
        )
    except BaseException as exn:
        # The one we want to catch is a pyo3_runtime.PanicException
        # But see https://github.com/PyO3/pyo3/issues/2880
        raise ProblemError("Solver failed to build problem") from exn

    result = solver.solve()

    solution = Solution(problem)
    solution.obj = result.obj_val
    solution.extras = {
        "s": result.s,
        "status": result.status,
        "solve_time": result.solve_time,
    }

    solution.found = result.status == clarabel.SolverStatus.Solved
    if not solution.found:
        warnings.warn(f"Clarabel.rs terminated with status {result.status}")

    solution.x = np.array(result.x)
    meq = A.shape[0] if A is not None else 0
    solution.y = result.z[:meq] if meq > 0 else np.empty((0,))
    if G is not None:
        z, z_box = split_dual_linear_box(np.array(result.z[meq:]), lb, ub)
        solution.z = z
        solution.z_box = z_box
    else:  # G is None
        solution.z = np.empty((0,))
        solution.z_box = np.empty((0,))
    return solution


def clarabel_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using Clarabel.rs.

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

    It is solved using `Clarabel.rs`_.

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
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = clarabel_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
