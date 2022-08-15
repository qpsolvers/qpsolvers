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
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    All other keyword arguments are forwarded as options to ProxQP. For
    instance, you can call ``proxqp_solve_qp(P, q, G, h, TODO(scaron): ...)``.
    See the solver documentation for details.
    """
    if initvals is not None:
        print("ProxQP: note that warm-start values ignored by wrapper")
    if lb is not None or ub is not None:
        # TODO(scaron): use native ProxQP bounds
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    A_prox = [] if A is None else A
    b_prox = [] if b is None else b
    C_prox = [] if G is None else G
    u_prox = [] if h is None else h
    l_prox = [] if h is None else np.full(h.shape, -np.infty)
    results = proxsuite.proxqp.dense.solve(
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
    found_solution = False  # TODO(scaron): this is a placeholder
    if not found_solution:
        # needs https://github.com/Simple-Robotics/proxsuite/issues/7
        return None
    return results.x
