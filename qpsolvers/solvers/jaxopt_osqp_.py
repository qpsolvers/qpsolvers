#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 Lev Kozlov

"""
Solver interface for `jaxopt's OSQP
<https://jaxopt.github.io/stable/quadratic_programming.html#osqpx>`__.

JAXopt is a library of hardware-accelerated, batchable and differentiable
optimizers implemented with JAX. JAX itself is a library for array-oriented
numerical computation that provides automatic differentiation and just-in-time
compilation.
"""

import warnings
from typing import Optional

import jax.numpy as jnp
import jaxopt
import numpy as np

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..problem import Problem
from ..solution import Solution


def jaxopt_osqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program with the OSQP algorithm implemented in jaxopt.

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
        Solution to the QP returned by the solver.

    Notes
    -----
    All other keyword arguments are forwarded as options to qpax. For
    instance, you can call ``qpax_solve_qp(P, q, G, h, solver_tol=1e-5)``.
    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``solver_tol``
         - Tolerance for the solver.

    Note that `jax` by default uses 32-bit floating point numbers, which can
    lead to numerical instability. If you encounter numerical issues, consider
    using 64-bit floating point numbers by setting
    ```python
    import jax
    jax.config.update("jax_enable_x64", True)
    ```
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]

    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by this wrapper")

    # construct the qpax problem
    G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
    if G is None:
        G = np.zeros((0, n))
        h = np.zeros((0,))

    osqp = jaxopt.OSQP()
    sol = osqp.run(
        params_obj=(jnp.array(P), jnp.array(q)),
        params_eq=(jnp.array(A), jnp.array(b)),
        params_ineq=(jnp.array(G), jnp.array(h)),
    ).params

    print(sol.primal)
    print(sol.dual_eq)
    print(sol.dual_ineq)
    solution = Solution(problem)
    solution.x = sol.primal
    solution.found = solution.state.status
    solution.y = sol.dual_eq

    # split the dual variables into
    # the box constraints and the linear constraints
    solution.z, solution.z_box = split_dual_linear_box(
        sol.dual_ineq, problem.lb, problem.ub
    )

    solution.extras = {
        "iter_num": solution.state.iter_num,
        "error": solution.state.error,
        "status": solution.state.status,
    }

    return solution


def qpax_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using qpax.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `qpax
    <https://github.com/kevin-tracy/qpax>`__.
    `Paper: <https://arxiv.org/pdf/2406.11749>`__.

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
    verbose :
        Set to `True` to print out extra information.
    initvals :
        Warm-start guess vector. Not used.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = jaxopt_osqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
