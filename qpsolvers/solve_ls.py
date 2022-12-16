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
Solve linear least squares.
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .solve_qp import solve_qp


def solve_ls(
    R: Union[np.ndarray, spa.csc_matrix],
    s: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    W: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    solver: Optional[str] = None,
    initvals: Optional[np.ndarray] = None,
    sym_proj: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Solve a constrained weighted linear Least Squares problem defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac12 \\| R x - s \\|^2_W
                = \\frac12 (R x - s)^T W (R x - s) \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using the QP solver selected by the ``solver`` keyword argument.

    Parameters
    ----------
    R :
        Union[np.ndarray, spa.csc_matrix] factor of the cost function (most
        solvers require :math:`R^T W R` to be positive definite, which means
        :math:`R` should have full row rank).
    s :
        Vector term of the cost function.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    A :
        Linear equality matrix.
    b :
        Linear equality vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    W :
        Definite symmetric weight matrix used to define the norm of the cost
        function. The standard L2 norm (W = Identity) is used by default.
    solver :
        Name of the QP solver, to choose in
        :data:`qpsolvers.available_solvers`. This argument is mandatory.
    initvals :
        Vector of initial `x` values used to warm-start the solver.
    sym_proj :
        Set to `True` when the `R` matrix provided is not symmetric.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Optimal solution if found, otherwise ``None``.

    Notes
    -----
    Extra keyword arguments given to this function are forwarded to the
    underlying solvers. For example, OSQP has a setting `eps_abs` which we can
    provide by ``solve_ls(R, s, G, h, solver='osqp', eps_abs=1e-4)``.
    """
    if sym_proj:
        R = 0.5 * (R + R.transpose())
        warnings.warn(
            "The `sym_proj` feature is deprecated "
            "and will be removed in qpsolvers v2.9",
            DeprecationWarning,
            stacklevel=2,
        )
    WR: Union[np.ndarray, spa.csc_matrix] = R if W is None else W @ R
    P = R.T @ WR
    q = -(s.T @ WR)
    if not isinstance(P, np.ndarray):
        P = P.tocsc()
    return solve_qp(
        P,
        q,
        G,
        h,
        A,
        b,
        lb,
        ub,
        solver=solver,
        initvals=initvals,
        sym_proj=False,
        verbose=verbose,
        **kwargs,
    )
