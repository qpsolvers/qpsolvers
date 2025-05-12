#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `OSQP <https://osqp.org/>`__.

The OSQP solver implements an operator-splitting method, more precisely an
alternating direction method of multipliers (ADMM). It is designed for both
dense and sparse problems, and convexity is the only assumption it makes on
problem data (for instance, it does not make any rank assumption, contrary to
solvers such as :ref:`CVXOPT <CVXOPT rank assumptions>` or :ref:`qpSWIFT
<qpSWIFT rank assumptions>`). If you are using OSQP in a scientific work,
consider citing the corresponding paper [Stellato2020]_.
"""

import warnings
from typing import Optional, Union

import numpy as np
import osqp
import scipy.sparse as spa
from osqp import OSQP
from scipy.sparse import csc_matrix

from ..conversions import ensure_sparse_matrices
from ..problem import Problem
from ..solution import Solution


def osqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using OSQP.

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
        Solution returned by the solver.

    Raises
    ------
    ValueError
        If the problem is clearly non-convex. See `this recommendation
        <https://osqp.org/docs/interfaces/status_values.html#status-values>`_.
        Note that OSQP may find the problem unfeasible if the problem is
        slightly non-convex (in this context, the meaning of "clearly" and
        "slightly" depends on how close the negative eigenvalues of :math:`P`
        are to zero).

    Note
    ----
    OSQP requires a symmetric `P` and won't check for errors otherwise. Check
    out this point if you `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.

    Notes
    -----
    Keyword arguments are forwarded to OSQP. For instance, we can call
    ``osqp_solve_qp(P, q, G, h, u, eps_abs=1e-8, eps_rel=0.0)``. OSQP settings
    include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iter``
         - Maximum number of iterations.
       * - ``time_limit``
         - Run time limit in seconds, 0 to disable.
       * - ``eps_abs``
         - Absolute feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_rel``
         - Relative feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_prim_inf``
         - Primal infeasibility tolerance.
       * - ``eps_dual_inf``
         - Dual infeasibility tolerance.
       * - ``polish``
         - Perform polishing. See `Polishing
           <https://osqp.org/docs/solver/#polishing>`_.

    Check out the `OSQP settings
    <https://osqp.org/docs/interfaces/solver_settings.html>`_ documentation for
    all available settings.

    Lower values for absolute or relative tolerances yield more precise
    solutions at the cost of computation time. See *e.g.* [Caron2022]_ for an
    overview of solver tolerances.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices("osqp", P, G, A)

    A_osqp = None
    l_osqp = None
    u_osqp = None
    if G is not None and h is not None:
        A_osqp = G
        l_osqp = np.full(h.shape, -np.inf)
        u_osqp = h
    if A is not None and b is not None:
        A_osqp = A if A_osqp is None else spa.vstack([A_osqp, A], format="csc")
        l_osqp = b if l_osqp is None else np.hstack([l_osqp, b])
        u_osqp = b if u_osqp is None else np.hstack([u_osqp, b])
    if lb is not None or ub is not None:
        lb = lb if lb is not None else np.full(q.shape, -np.inf)
        ub = ub if ub is not None else np.full(q.shape, +np.inf)
        E = spa.eye(q.shape[0])
        A_osqp = E if A_osqp is None else spa.vstack([A_osqp, E], format="csc")
        l_osqp = lb if l_osqp is None else np.hstack([l_osqp, lb])
        u_osqp = ub if u_osqp is None else np.hstack([u_osqp, ub])

    kwargs["verbose"] = verbose
    solver = OSQP()
    solver.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, **kwargs)
    if initvals is not None:
        solver.warm_start(x=initvals)

    res = solver.solve()
    success_status = osqp.constant("OSQP_SOLVED")

    solution = Solution(problem)
    solution.extras = {
        "info": res.info,
        "dual_inf_cert": res.dual_inf_cert,
        "prim_inf_cert": res.prim_inf_cert,
    }

    solution.found = res.info.status_val == success_status
    if not solution.found:
        warnings.warn(f"OSQP exited with status '{res.info.status}'")
    solution.x = res.x
    m = G.shape[0] if G is not None else 0
    meq = A.shape[0] if A is not None else 0
    solution.z = res.y[:m] if G is not None else np.empty((0,))
    solution.y = res.y[m : m + meq] if A is not None else np.empty((0,))
    solution.z_box = (
        res.y[m + meq :]
        if lb is not None or ub is not None
        else np.empty((0,))
    )
    return solution


def osqp_solve_qp(
    P: Union[np.ndarray, csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using OSQP.

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

    It is solved using `OSQP <https://github.com/oxfordcontrol/osqp>`__.

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
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ValueError
        If the problem is clearly non-convex. See `this recommendation
        <https://osqp.org/docs/interfaces/status_values.html#status-values>`_.
        Note that OSQP may find the problem unfeasible if the problem is
        slightly non-convex (in this context, the meaning of "clearly" and
        "slightly" depends on how close the negative eigenvalues of :math:`P`
        are to zero).

    Note
    ----
    OSQP requires a symmetric `P` and won't check for errors otherwise. Check
    out this point if you `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.

    Notes
    -----
    Keyword arguments are forwarded to OSQP. For instance, we can call
    ``osqp_solve_qp(P, q, G, h, u, eps_abs=1e-8, eps_rel=0.0)``. OSQP settings
    include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iter``
         - Maximum number of iterations.
       * - ``time_limit``
         - Run time limit in seconds, 0 to disable.
       * - ``eps_abs``
         - Absolute feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_rel``
         - Relative feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_prim_inf``
         - Primal infeasibility tolerance.
       * - ``eps_dual_inf``
         - Dual infeasibility tolerance.
       * - ``polish``
         - Perform polishing. See `Polishing
           <https://osqp.org/docs/solver/#polishing>`_.

    Check out the `OSQP settings
    <https://osqp.org/docs/interfaces/solver_settings.html>`_ documentation for
    all available settings.

    Lower values for absolute or relative tolerances yield more precise
    solutions at the cost of computation time. See *e.g.* [Caron2022]_ for an
    overview of solver tolerances.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = osqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
