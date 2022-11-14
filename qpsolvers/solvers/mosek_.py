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

"""Solver interface for MOSEK"""

from typing import Optional, Union

import cvxopt.msk
import mosek
from numpy import ndarray
from scipy.sparse import csc_matrix

from .cvxopt_ import cvxopt_solve_qp


def mosek_solve_qp(
    P: Union[ndarray, csc_matrix],
    q: ndarray,
    G: Union[ndarray, csc_matrix],
    h: ndarray,
    A: Optional[Union[ndarray, csc_matrix]] = None,
    b: Optional[ndarray] = None,
    lb: Optional[ndarray] = None,
    ub: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
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

    using the `MOSEK interface from CVXOPT
    <https://cvxopt.org/userguide/coneprog.html#optional-solvers>`_.

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

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
    """
    original_options = cvxopt.solvers.options
    cvxopt.solvers.options["mosek"] = {mosek.iparam.log: 1 if verbose else 0}
    x = cvxopt_solve_qp(P, q, G, h, A, b, lb, ub, "mosek", initvals, **kwargs)
    cvxopt.solvers.options = original_options
    return x
