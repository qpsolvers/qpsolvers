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
Solver interface for `ProxQP
<https://github.com/Simple-Robotics/proxsuite#proxqp>`__.

ProxQP is the QP solver from ProxSuite, a collection of open-source solvers
rooted in revisited primal-dual proximal algorithms. If you use ProxQP in some
academic work, consider citing the corresponding paper [Bambade2022]_.
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
from proxsuite import proxqp

from ..problem import Problem
from ..solution import Solution


def __combine_inequalities(G, h, lb, ub, n: int, use_csc: bool):
    """
    Combine linear and box inequalities for ProxQP.

    Parameters
    ----------
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    n :
        Number of optimization variables.
    use_csc :
        If ``True``, use sparse rather than dense matrices.

    Returns
    -------
    :
        Linear inequality matrices :math:`C`, :math:`l` and :math:`u`.
    """
    if lb is None and ub is None:
        C_prox = G
        u_prox = h
        l_prox = np.full(h.shape, -np.infty) if h is not None else None
    elif G is None:
        # lb is not None or ub is not None:
        C_prox = spa.eye(n, format="csc") if use_csc else np.eye(n)
        u_prox = ub
        l_prox = lb
    elif h is not None:
        # G is not None and h is not None and not (lb is None and ub is None)
        C_prox = (
            spa.vstack((G, spa.eye(n)), format="csc")
            if use_csc
            else np.vstack((G, np.eye(n)))
        )
        ub = ub if ub is not None else np.full(G.shape[1], +np.infty)
        lb = lb if lb is not None else np.full(G.shape[1], -np.infty)
        l_prox = np.hstack((np.full(h.shape, -np.infty), lb))
        u_prox = np.hstack((h, ub))
    else:  # G is not None and h is None
        raise ValueError("Inconsistent inequalities: G is set but h is None")
    return C_prox, u_prox, l_prox


def __select_backend(backend: Optional[str], use_csc: bool):
    """
    Select backend function for ProxQP.

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
    """
    if backend is None:
        return proxqp.sparse.solve if use_csc else proxqp.dense.solve
    if backend == "dense":
        return proxqp.dense.solve
    if backend == "sparse":
        return proxqp.sparse.solve
    raise ValueError(f'Unknown ProxQP backend "{backend}')


def proxqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Solution:
    """
    Solve a Quadratic Program using `ProxQP
    <https://github.com/Simple-Robotics/proxsuite>`__.

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
           [tolerances]_ for an overview of solver tolerances.
       * - ``eps_rel``
         - Relative stopping criterion of the solver. See *e.g.* [tolerances]_
           for an overview of solver tolerances.
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
            raise ValueError(
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
    C_prox, u_prox, l_prox = __combine_inequalities(G, h, lb, ub, n, use_csc)
    solve_function = __select_backend(backend, use_csc)
    result = solve_function(
        P,
        q,
        A,
        b,
        C_prox,
        l_prox,
        u_prox,
        verbose=verbose,
        **kwargs,
    )
    solution = Solution(problem)
    solution.extras = {"info": result.info}
    if result.info.status != proxqp.QPSolverOutput.PROXQP_SOLVED:
        return solution
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
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\underset{\\mbox{minimize}}{x} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b                    \\\\
            & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `ProxQP <https://github.com/Simple-Robotics/proxsuite>`__.

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
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = proxqp_solve_problem(
        problem, initvals, verbose, backend, **kwargs
    )
    return solution.x
