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
Solver interface for `OSQP <https://osqp.org/>`__.

The OSQP solver implements an Operator-Splitting method for large-scale convex
quadratic programming. It is designed for both dense and sparse problems, and
convexity is the only assumption it makes on problem data (for instance, it
does not make any rank assumption contrary to :ref:`CVXOPT <CVXOPT rank
assumptions>` or :ref:`qpSWIFT <qpSWIFT rank assumptions>`). If you are using
OSQP in some academic work, consider citing the corresponding paper
[Stellato2020]_.
"""

import warnings
from typing import Optional, Union

import numpy as np
import osqp
import scipy.sparse as spa
from osqp import OSQP
from scipy.sparse import csc_matrix

from ..conversions import warn_about_sparse_conversion
from ..problem import Problem
from ..solution import Solution


def osqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """
    Solve a quadratic program using `OSQP
    <https://github.com/oxfordcontrol/osqp>`__.

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
    solutions at the cost of computation time. See *e.g.* [tolerances]_ for an
    overview of solver tolerances.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if isinstance(P, np.ndarray):
        warn_about_sparse_conversion("P")
        P = csc_matrix(P)
    if isinstance(G, np.ndarray):
        warn_about_sparse_conversion("G")
        G = csc_matrix(G)
    if isinstance(A, np.ndarray):
        warn_about_sparse_conversion("A")
        A = csc_matrix(A)

    A_osqp = None
    l_osqp = None
    u_osqp = None
    if G is not None and h is not None:
        A_osqp = G
        l_osqp = np.full(h.shape, -np.infty)
        u_osqp = h
    if A is not None and b is not None:
        A_osqp = A if A_osqp is None else spa.vstack([A_osqp, A], format="csc")
        l_osqp = b if l_osqp is None else np.hstack([l_osqp, b])
        u_osqp = b if u_osqp is None else np.hstack([u_osqp, b])
    if lb is not None or ub is not None:
        lb = lb if lb is not None else np.full(q.shape, -np.infty)
        ub = ub if ub is not None else np.full(q.shape, +np.infty)
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
    if res.info.status_val != success_status:
        warnings.warn(f"OSQP exited with status '{res.info.status}'")
        return solution
    solution.x = res.x
    m = G.shape[0] if G is not None else 0
    meq = A.shape[0] if A is not None else 0
    if G is not None:
        solution.z = res.y[:m]
    if A is not None:
        solution.y = res.y[m : m + meq]
    if lb is not None or ub is not None:
        solution.z_box = res.y[m + meq :]
    solution.extras = {
        "dual_inf_cert": res.dual_inf_cert,
        "info": res.info,
        "prim_inf_cert": res.prim_inf_cert,
    }
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
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b
        \\end{array}\\end{split}

    using `OSQP <https://github.com/oxfordcontrol/osqp>`_.

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
    solutions at the cost of computation time. See *e.g.* [tolerances]_ for an
    overview of solver tolerances.
    """
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = osqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x
