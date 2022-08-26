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
Solver interface for `ProxQP <https://github.com/qpSWIFT/qpSWIFT>`__.

ProxQP is the QP solver from ProxSuite, a collection of open-source solvers
rooted in revisited primal-dual proximal algorithms. If you are using ProxQP in
your work, consider citing the corresponding paper: `PROX-QP: Yet another
Quadratic Programming Solver for Robotics and beyond
<https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/>`__.
"""

from typing import Optional, Union

import numpy as np
import proxsuite
import scipy.sparse as spa

from .conversions import linear_from_box_inequalities


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
        \\mbox{minimize} &
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
        Positive semidefinite quadratic-cost matrix.
    q :
        Quadratic-cost vector.
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
        Solution to the QP, if found, otherwise ``None``.

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
           this is a laxer default than other solvers).
       * - ``eps_rel``
         - Relative stopping criterion of the solver.
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
        # TODO(scaron): forward warm-start values
        print("ProxQP: note that warm-start values ignored by wrapper")
    if lb is not None or ub is not None:
        # TODO(scaron): use native ProxQP bounds
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    # TODO(scaron): https://github.com/Simple-Robotics/proxsuite/issues/6
    n = P.shape[1]
    A_prox = np.zeros((0, n)) if A is None else A
    b_prox = np.zeros(0) if b is None else b
    C_prox = np.zeros((0, n)) if G is None else G
    u_prox = np.zeros(0) if h is None else h
    l_prox = np.zeros(0) if h is None else np.full(h.shape, -np.infty)
    if backend is None:
        if isinstance(P, np.ndarray):
            solve_function = proxsuite.proxqp.dense.solve
        else:  # isinstance(P, spa.csc_matrix):
            solve_function = proxsuite.proxqp.sparse.solve
    elif backend == "dense":
        solve_function = proxsuite.proxqp.dense.solve
    elif backend == "sparse":
        solve_function = proxsuite.proxqp.sparse.solve
    else:  # invalid argument
        raise ValueError(f'Unknown ProxQP backend "{backend}')
    result = solve_function(
        P,
        q,
        A_prox,
        b_prox,
        C_prox,
        u_prox,
        l_prox,
        verbose=verbose,
        **kwargs,
    )
    if result.info.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
        return None
    return result.x
