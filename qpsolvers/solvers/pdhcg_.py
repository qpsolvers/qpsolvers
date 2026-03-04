#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for PDHCG.

PDHCG (Primal-Dual Hybrid Conjugate Gradient) is a high-performance, 
GPU-accelerated solver designed for large-scale convex Quadratic Programming (QP).
It is particularly efficient for huge-scale problems by fully 
leveraging NVIDIA CUDA architectures.

Note:
    To use this solver, you need an NVIDIA GPU and the ``pdhcg`` package 
    installed via ``pip install pdhcg``. For advanced installation (e.g., 
    custom CUDA paths), please refer to the 
    `official PDHCG-II repository <https://github.com/Lhongpei/PDHCG-II>`_.

References:
    - `PDHCG-II: An Enhanced Version of PDHCG for Large-Scale Convex QP <https://arxiv.org/abs/2602.23967>`_
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from ..problem import Problem
from ..solution import Solution

from pdhcg import Model


def pdhcg_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using PDHCG.
    
    The quadratic program is defined as:

        minimize      1/2 x^T P x + q^T x
        subject to    G x <= h
                      A x = b
                      lb <= x <= ub

    Parameters
    ----------
    P :
        Symmetric quadratic cost matrix.
    q :
        Quadratic cost vector.
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
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Optimal primal solution if found, otherwise `None`.

    Notes
    -----
    Keyword arguments are forwarded to PDHCG as solver parameters. For instance,
    you can call ``pdhcg_solve_qp(..., TimeLimit=60, OptimalityTol=1e-5)``. 
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
         - Relative feasibility tolerance for primal/dual residuals (default: 1e-4).
       * - ``OutputFlag``
         - Enable (True) or disable (False) console logging output.

    For a comprehensive list of advanced scaling and restart parameters, please refer to the
    `PDHCG Python Interface Documentation <https://github.com/Lhongpei/PDHCG-II/tree/main/python>`_.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()

    C_mats = []
    l_bounds = []
    u_bounds = []

    if G is not None:
        C_mats.append(G)
        l_bounds.append(np.full(h.shape, -np.inf))
        u_bounds.append(h)

    if A is not None:
        C_mats.append(A)
        l_bounds.append(b)
        u_bounds.append(b)

    if C_mats:
        if any(spa.issparse(mat) for mat in C_mats):
            constraint_matrix = spa.vstack(C_mats, format="csr")
        else:
            constraint_matrix = np.vstack(C_mats)
        constraint_lower_bound = np.concatenate(l_bounds)
        constraint_upper_bound = np.concatenate(u_bounds)
    else:
        constraint_matrix = None
        constraint_lower_bound = None
        constraint_upper_bound = None

    model = Model(
        objective_matrix=P,
        objective_vector=q,
        constraint_matrix=constraint_matrix,
        constraint_lower_bound=constraint_lower_bound,
        constraint_upper_bound=constraint_upper_bound,
        variable_lower_bound=lb,
        variable_upper_bound=ub
    )

    if verbose: model.setParam("OutputFlag", 1)
    if kwargs:
        model.setParams(**kwargs)

    if initvals is not None:
        model.setWarmStart(primal=initvals)

    model.optimize()

    solution = Solution(problem)
    
    status_str = str(model.Status).upper() if model.Status else ""
    solution.found = (status_str == "OPTIMAL")  
    
    if solution.found and model.X is not None:
        solution.x = np.array(model.X)
        solution.obj = model.ObjVal

    solution.runtime = model.Runtime
    solution.iter = model.IterCount

    if solution.found and model.Pi is not None and C_mats:
        pi = np.array(model.Pi)
        idx = 0
        if G is not None:
            num_g = G.shape[0]
            solution.z = -pi[idx : idx + num_g]
            idx += num_g
        else:
            solution.z = np.empty((0,))

        if A is not None:
            num_a = A.shape[0]
            solution.y = -pi[idx : idx + num_a]
        else:
            solution.y = np.empty((0,))
    else:
        solution.z = np.empty((0,)) if G is None else np.empty(G.shape[0])
        solution.y = np.empty((0,)) if A is None else np.empty(A.shape[0])

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
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using HiGHS.

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

    It is solved using `HiGHS <https://github.com/ERGO-Code/HiGHS>`__.

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

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
        
    Notes
    -----
    Keyword arguments are forwarded to PDHCG as solver parameters. For instance,
    you can call ``pdhcg_solve_qp(..., TimeLimit=60, OptimalityTol=1e-5)``. 
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
         - Relative feasibility tolerance for primal/dual residuals (default: 1e-4).
       * - ``OutputFlag``
         - Enable (True) or disable (False) console logging output.

    For a comprehensive list of advanced scaling and restart parameters, please refer to the
    `PDHCG Python Interface Documentation <https://github.com/Lhongpei/PDHCG-II/tree/main/python>`_.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = pdhcg_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None