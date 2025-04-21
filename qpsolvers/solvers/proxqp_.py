#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `ProxQP`_.

.. _ProxQP: https://github.com/Simple-Robotics/proxsuite#proxqp

ProxQP is a primal-dual augmented Lagrangian method with proximal heuristics.
It converges to the solution of feasible problems, or to the solution to the
closest feasible one if the input problem is unfeasible. ProxQP is part of the
ProxSuite collection of open-source solvers. If you use ProxQP in a scientific
work, consider citing the corresponding paper [Bambade2022]_.
"""

from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
from proxsuite import proxqp

from ..conversions import combine_linear_box_inequalities
from ..exceptions import ParamError
from ..problem import Problem
from ..solution import Solution


def __select_backend(backend: Optional[str], use_csc: bool):
    """Select backend function for ProxQP.

    Parameters
    ----------
    backend :
        ProxQP backend to use in ``[None, "dense", "sparse"]``. If ``None``
        (default), the backend is selected based on the type of ``P``.
    use_csc :
        If ``True``, use sparse matrices if the backend is not specified.

    Returns
    -------
    :
        Backend solve function.

    Raises
    ------
    ParamError
        If the required backend is not a valid ProxQP backend.
    """
    if backend is None:
        return proxqp.sparse.solve if use_csc else proxqp.dense.solve
    if backend == "dense":
        return proxqp.dense.solve
    if backend == "sparse":
        return proxqp.sparse.solve
    raise ParamError(f'Unknown ProxQP backend "{backend}')


def proxqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using ProxQP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    backend :
        ProxQP backend to use in ``[None, "dense", "sparse"]``. If ``None``
        (default), the backend is selected based on the type of ``P``.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Raises
    ------
    ParamError
        If a warm-start value is given both in `initvals` and the `x` keyword
        argument.

    Notes
    -----
    All other keyword arguments are forwarded as options to ProxQP. For
    instance, you can call ``proxqp_solve_qp(P, q, G, h, eps_abs=1e-6)``.
    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``x``
         - Warm start value for the primal variable.
       * - ``y``
         - Warm start value for the dual Lagrange multiplier for equality
           constraints.
       * - ``z``
         - Warm start value for the dual Lagrange multiplier for inequality
           constraints.
       * - ``eps_abs``
         - Asbolute stopping criterion of the solver (default: 1e-3, note that
           this is a laxer default than other solvers). See *e.g.*
           [Caron2022]_ for an overview of solver tolerances.
       * - ``eps_rel``
         - Relative stopping criterion of the solver. See *e.g.* [Caron2022]_
           for an overview of solver tolerances.
       * - ``check_duality_gap``
         - If set to true (false by default), ProxQP will include the duality
           gap in absolute and relative stopping criteria.
       * - ``mu_eq``
         - Proximal step size wrt equality constraints multiplier.
       * - ``mu_in``
         - Proximal step size wrt inequality constraints multiplier.
       * - ``rho``
         - Proximal step size wrt primal variable.
       * - ``compute_preconditioner``
         - If ``True`` (default), the preconditioner will be derived.
       * - ``compute_timings``
         - If ``True`` (default), timings will be computed by the solver (setup
           time, solving time, and run time = setup time + solving time).
       * - ``max_iter``
         - Maximal number of authorized outer iterations.
       * - ``initial_guess``
         - Sets the initial guess option for initilizing x, y and z.

    This list is not exhaustive. Check out the `solver documentation
    <https://simple-robotics.github.io/proxsuite/>`__ for details.
    """
    if initvals is not None:
        if "x" in kwargs:
            raise ParamError(
                "Warm-start value specified in both `initvals` and `x` kwargs"
            )
        kwargs["x"] = initvals
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]
    use_csc: bool = (
        not isinstance(P, np.ndarray)
        or (G is not None and not isinstance(G, np.ndarray))
        or (A is not None and not isinstance(A, np.ndarray))
    )
    Cx, ux, lx = combine_linear_box_inequalities(G, h, lb, ub, n, use_csc)
    solve = __select_backend(backend, use_csc)
    result = solve(
        P,
        q,
        A,
        b,
        Cx,
        lx,
        ux,
        verbose=verbose,
        **kwargs,
    )
    solution = Solution(problem)
    solution.extras = {"info": result.info}
    solution.found = result.info.status == proxqp.QPSolverOutput.PROXQP_SOLVED
    solution.x = result.x
    solution.y = result.y
    if lb is not None or ub is not None:
        solution.z = result.z[:-n]
        solution.z_box = result.z[-n:]
    else:  # lb is None and ub is None
        solution.z = result.z
    return solution


def proxqp_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    lb: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    ub: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using ProxQP.

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

    It is solved using `ProxQP
    <https://github.com/Simple-Robotics/proxsuite#proxqp>`__.

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
        Warm-start guess vector.
    backend :
        ProxQP backend to use in ``[None, "dense", "sparse"]``. If ``None``
        (default), the backend is selected based on the type of ``P``.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = proxqp_solve_problem(
        problem, initvals, verbose, backend, **kwargs
    )
    return solution.x if solution.found else None
