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

"""Solver interface for qpOASES"""

from typing import Optional

import numpy as np
from numpy import array, hstack, ones, vstack, zeros
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyQProblem as QProblem
from qpoases import PyQProblemB as QProblemB
from qpoases import PyReturnValue as ReturnValue

from .conversions import linear_from_box_inequalities


__infty__ = 1e10
__options__ = Options()
__options__.printLevel = PrintLevel.NONE


def qpoases_solve_qp(
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
    max_wsr: int = 1000,
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

    using `qpOASES <https://github.com/coin-or/qpOASES>`__.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix.
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
    max_wsr :
        Maximum number of Working-Set Recalculations given to qpOASES.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    This function relies on some updates from the standard distribution of
    qpOASES. See the `installation instructions
    <https://scaron.info/doc/qpsolvers/installation.html#qpoases>`_ for
    details.

    Empty bounds (`lb`, `ub`, `lbA` or `ubA`) are allowed. This is possible in
    the C++ API but not by the Python API of qpOASES (as of version 3.2.0). If
    using them, be sure to update the Cython file (`qpoases.pyx`) in your
    distribution of qpOASES to convert ``None`` to the null pointer.
    """
    if initvals is not None:
        print("qpOASES: note that warm-start values ignored by wrapper")
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    n = P.shape[0]
    lb = None  # TODO(scaron): use native qpOASES box bounds
    ub = None  # TODO(scaron): idem
    lb_C: Optional[np.ndarray] = None
    has_constraint = True
    if G is not None and h is not None:
        if A is not None and b is not None:
            C = vstack([G, A, A])
            lb_C = hstack([-__infty__ * ones(h.shape[0]), b, b])
            ub_C = hstack([h, b, b])
        else:  # no equality constraint
            C = G
            ub_C = h
    else:  # no inequality constraint
        if A is not None and b is not None:
            C = A
            lb_C = b
            ub_C = b
        else:  # no equality constraint either
            has_constraint = False
    __options__.printLevel = PrintLevel.MEDIUM if verbose else PrintLevel.NONE
    if has_constraint:
        qp = QProblem(n, C.shape[0])
        qp.setOptions(__options__)
        return_value = qp.init(P, q, C, lb, ub, lb_C, ub_C, array([max_wsr]))
        if return_value == ReturnValue.MAX_NWSR_REACHED:
            print(f"qpOASES reached the maximum number of WSR ({max_wsr})")
    else:  # no constraint
        qp = QProblemB(n)
        qp.setOptions(__options__)
        qp.init(P, q, lb, ub, max_wsr)
    x_opt = zeros(n)
    ret = qp.getPrimalSolution(x_opt)
    if ret != 0:  # 0 == SUCCESSFUL_RETURN code of qpOASES
        print(f"qpOASES failed with return code {ret}")
    return x_opt
