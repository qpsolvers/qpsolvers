#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `HPIPM <https://github.com/giaf/hpipm>`__.

HPIPM is a high-performance interior point method for solving convex quadratic
programs. It is designed to be efficient for small to medium-size problems
arising in model predictive control and embedded optmization. If you are using
HPIPM in a scientific work, consider citing the corresponding paper
[Frison2020]_.
"""

import warnings
from typing import Optional

import hpipm_python.common as hpipm
import numpy as np

from ..problem import Problem
from ..solution import Solution


def hpipm_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    mode: str = "balance",
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using HPIPM.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    mode :
        Solver mode, which provides a set of default solver arguments. Pick one
        of ["speed_abs", "speed", "balance", "robust"]. These modes are
        documented in section 4.2 *IPM implementation choices* of the reference
        paper [Frison2020]_. The default one is "balance".
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to HPIPM. For instance, we can call
    ``hpipm_solve_qp(P, q, G, h, u, tol_eq=1e-5)``. HPIPM settings include the
    following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``iter_max``
         - Maximum number of iterations.
       * - ``tol_eq``
         - Equality constraint tolerance.
       * - ``tol_ineq``
         - Inequality constraint tolerance.
       * - ``tol_comp``
         - Complementarity condition tolerance.
       * - ``tol_dual_gap``
         - Duality gap tolerance.
       * - ``tol_stat``
         - Stationarity condition tolerance.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if verbose:
        warnings.warn("verbose keyword argument is ignored by HPIPM")

    # setup the problem dimensions
    nv = q.shape[0]
    ne = b.shape[0] if b is not None else 0
    ng = h.shape[0] if h is not None else 0

    nlb = lb.shape[0] if lb is not None else 0
    nub = ub.shape[0] if ub is not None else 0
    nb = max(nlb, nub)

    dim = hpipm.hpipm_dense_qp_dim()
    dim.set("nv", nv)
    dim.set("nb", nb)
    dim.set("ne", ne)
    dim.set("ng", ng)

    # setup the problem data
    qp = hpipm.hpipm_dense_qp(dim)
    qp.set("H", P)
    qp.set("g", q)

    if ng > 0:
        qp.set("C", G)
        qp.set("ug", h)
        # mask out the lower bound
        qp.set("lg_mask", np.zeros_like(h, dtype=bool))

    if ne > 0:
        qp.set("A", A)
        qp.set("b", b)

    if nb > 0:
        # mark all variables as box-constrained
        qp.set("idxb", np.arange(nv))

        # need to mask out lb or ub if the box constraints are only one-sided
        # we also mask out infinities (and set the now-irrelevant value to
        # zero), since HPIPM doesn't like infinities
        if nlb > 0 and lb is not None:  # help mypy
            lb_mask = np.isinf(lb)
            lb[lb_mask] = 0.0
            qp.set("lb", lb)
            qp.set("lb_mask", ~lb_mask)
        else:
            qp.set("lb_mask", np.zeros(nb, dtype=bool))

        if nub > 0 and ub is not None:  # help mypy
            ub_mask = np.isinf(ub)
            ub[ub_mask] = 0.0
            qp.set("ub", ub)
            qp.set("ub_mask", ~ub_mask)
        else:
            qp.set("ub_mask", np.zeros(nb, dtype=bool))

    solver_args = hpipm.hpipm_dense_qp_solver_arg(dim, mode)
    for key, val in kwargs.items():
        solver_args.set(key, val)

    sol = hpipm.hpipm_dense_qp_sol(dim)
    if initvals is not None:
        solver_args.set("warm_start", 1)
        sol.set("v", initvals)

    solver = hpipm.hpipm_dense_qp_solver(dim, solver_args)
    solver.solve(qp, sol)

    status = solver.get("status")

    solution = Solution(problem)
    solution.extras = {
        "status": status,
        "max_res_stat": solver.get("max_res_stat"),
        "max_res_eq": solver.get("max_res_eq"),
        "max_res_ineq": solver.get("max_res_ineq"),
        "max_res_comp": solver.get("max_res_comp"),
        "iter": solver.get("iter"),
        "stat": solver.get("stat"),
    }
    solution.found = status == 0
    if not solution.found:
        warnings.warn(f"HPIPM exited with status '{status}'")

    # the equality multipliers in HPIPM are opposite in sign compared to what
    # we expect here
    solution.x = sol.get("v").flatten()
    solution.y = -sol.get("pi").flatten() if ne > 0 else np.empty((0,))
    solution.z = sol.get("lam_ug").flatten() if ng > 0 else np.empty((0,))
    if nb > 0:
        solution.z_box = (sol.get("lam_ub") - sol.get("lam_lb")).flatten()
    else:
        solution.z_box = np.empty((0,))
    return solution


def hpipm_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    mode: str = "balance",
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using HPIPM.

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

    It is solved using `HPIPM <https://github.com/giaf/hpipm>`__.

    Parameters
    ----------
    P :
        Symmetric cost matrix.
    q :
        Cost vector.
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
        Warm-start guess vector.
    mode :
        Solver mode, which provides a set of default solver arguments. Pick one
        of ["speed_abs", "speed", "balance", "robust"]. Default is "balance".
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to HPIPM. For instance, we can call
    ``hpipm_solve_qp(P, q, G, h, u, tol_eq=1e-5)``. HPIPM settings include the
    following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``iter_max``
         - Maximum number of iterations.
       * - ``tol_eq``
         - Equality constraint tolerance.
       * - ``tol_ineq``
         - Inequality constraint tolerance.
       * - ``tol_comp``
         - Complementarity condition tolerance.
       * - ``tol_dual_gap``
         - Duality gap tolerance.
       * - ``tol_stat``
         - Stationarity condition tolerance.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = hpipm_solve_problem(problem, initvals, mode, verbose, **kwargs)
    return solution.x if solution.found else None
