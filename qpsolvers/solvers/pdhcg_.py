#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Solver interface for PDHCG.

PDHCG (Primal-Dual Hybrid Conjugate Gradient) solves large-scale convex
quadratic programs on CPU or CUDA. Install the ``pdhcg`` package with the
backends you need; see the
`PDHCG installation guide <https://pdhcg.github.io/PDHCG/installation/>`_.

References
----------
- `PDHCG-II <https://arxiv.org/abs/2602.23967>`_
"""

import time
from typing import Any, List, Optional, Union

import numpy as np
import scipy.sparse as spa
from pdhcg import Model

from ..problem import Problem
from ..solution import Solution


def pdhcg_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    *,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Solution:
    r"""Solve a quadratic program using PDHCG.

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

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.
    device :
        Compiled PDHCG backend, such as ``"cpu"`` or ``"cuda"``. If omitted,
        use the default selected when PDHCG was built.

    Returns
    -------
    :
        Solution with primal and dual values when an optimum is found.

    Notes
    -----
    ``device`` is passed to ``Model.optimize``; other keyword arguments are
    forwarded as solver parameters. For example, call
    ``pdhcg_solve_qp(..., device="cpu", Threads=4, TimeLimit=60)``.
    Build PDHCG with the requested backend; ``Threads`` controls CPU solves.
    Common PDHCG parameters include:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``TimeLimit``
         - Maximum wall-clock time in seconds (default: 3600.0).
       * - ``IterationLimit``
         - Maximum number of iterations.
       * - ``OptimalityTol``
         - Relative tolerance for optimality gap (default: 1e-4).
       * - ``FeasibilityTol``
         - Relative feasibility tolerance for residuals (default: 1e-4).
       * - ``OutputFlag``
         - Enable (True) or disable (False) console logging output.

    See the `PDHCG parameter reference
    <https://pdhcg.github.io/PDHCG/python/parameters/>`_ for more options.
    """
    build_start_time = time.perf_counter()
    P, q, G, h, A, b, lb, ub = problem.unpack()

    C_mats: List[Any] = []
    l_bounds: List[Any] = []
    u_bounds: List[Any] = []

    if G is not None and h is not None:
        C_mats.append(G)
        l_bounds.append(np.full(h.shape, -np.inf))
        u_bounds.append(h)

    if A is not None and b is not None:
        C_mats.append(A)
        l_bounds.append(b)
        u_bounds.append(b)

    constraint_matrix: Optional[
        Union[np.ndarray, spa.csr_matrix, spa.csc_matrix]
    ] = None
    constraint_lower_bound: Optional[np.ndarray] = None
    constraint_upper_bound: Optional[np.ndarray] = None

    if C_mats:
        if any(spa.issparse(mat) for mat in C_mats):
            constraint_matrix = spa.vstack(C_mats, format="csr")  # type: ignore
        else:
            constraint_matrix = np.vstack(C_mats)  # type: ignore
        constraint_lower_bound = np.concatenate(l_bounds)  # type: ignore
        constraint_upper_bound = np.concatenate(u_bounds)  # type: ignore

    model = Model(
        objective_matrix=P,
        objective_vector=q,
        constraint_matrix=constraint_matrix,
        constraint_lower_bound=constraint_lower_bound,
        constraint_upper_bound=constraint_upper_bound,
        variable_lower_bound=lb,
        variable_upper_bound=ub,
    )

    if verbose:
        model.setParam("OutputFlag", 2)
    else:
        model.setParam("OutputFlag", 0)

    if kwargs:
        model.setParams(**kwargs)

    if initvals is not None:
        model.setWarmStart(primal=initvals)

    solve_start_time = time.perf_counter()
    if device is None:
        model.optimize()
    else:
        model.optimize(device=device)
    solve_end_time = time.perf_counter()

    solution = Solution(problem)
    solution.build_time = solve_start_time - build_start_time
    solution.solve_time = solve_end_time - solve_start_time

    status_str = str(model.Status).upper() if model.Status else ""
    solution.found = status_str == "OPTIMAL"

    if solution.found and model.X is not None:
        solution.x = np.array(model.X)
        solution.obj = model.ObjVal

    solution.extras["runtime"] = model.Runtime
    solution.extras["iter"] = model.IterCount
    solution.extras["status"] = status_str

    if solution.found and solution.x is not None:
        num_g = G.shape[0] if G is not None else 0
        pi = np.asarray(model.Pi)
        solution.z = -pi[:num_g] if G is not None else np.empty(0)
        solution.y = -pi[num_g:] if A is not None else np.empty(0)
        if lb is not None or ub is not None:
            # PDHCG exposes row multipliers; recover bound multipliers from
            # stationarity: P x + q + G.T z + A.T y + z_box = 0.
            solution.z_box = -(P @ solution.x + q)
            if G is not None:
                solution.z_box -= G.T @ solution.z
            if A is not None:
                solution.z_box -= A.T @ solution.y
            # Missing bounds cannot carry a multiplier.
            lower = np.isfinite(lb) if lb is not None else False
            upper = np.isfinite(ub) if ub is not None else False
            solution.z_box = np.where(
                solution.z_box < 0.0,
                np.where(lower, solution.z_box, 0.0),
                np.where(upper, solution.z_box, 0.0),
            )
        else:
            solution.z_box = np.empty(0)

    return solution


def pdhcg_solve_qp(
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
    *,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using PDHCG.

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

    It is solved using `PDHCG <https://github.com/pdhcg/PDHCG>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
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
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.
    device :
        Compiled PDHCG backend, such as ``"cpu"`` or ``"cuda"``. If omitted,
        use the default selected when PDHCG was built.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    ``device`` is passed to ``Model.optimize``; other keyword arguments are
    forwarded as solver parameters. For example, call
    ``pdhcg_solve_qp(..., device="cpu", Threads=4, TimeLimit=60)``.
    Build PDHCG with the requested backend; ``Threads`` controls CPU solves.
    Common PDHCG parameters include:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``TimeLimit``
         - Maximum wall-clock time in seconds (default: 3600.0).
       * - ``IterationLimit``
         - Maximum number of iterations.
       * - ``OptimalityTol``
         - Relative tolerance for optimality gap (default: 1e-4).
       * - ``FeasibilityTol``
         - Relative feasibility tolerance for residuals (default: 1e-4).
       * - ``OutputFlag``
         - Enable (True) or disable (False) console logging output.

    See the `PDHCG parameter reference
    <https://pdhcg.github.io/PDHCG/python/parameters/>`_ for more options.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = pdhcg_solve_problem(
        problem, initvals, verbose, device=device, **kwargs
    )
    return solution.x if solution.found else None
