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

"""Solve linear least squares."""

from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .problem import Problem
from .solve_qp import solve_qp


def __solve_dense_ls(
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
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
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
        verbose=verbose,
        **kwargs,
    )


def __solve_sparse_ls(
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
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    m, n = R.shape
    eye_m = spa.eye(m, format="csc")
    P = spa.block_diag(
        [spa.csc_matrix((n, n)), eye_m if W is None else W], format="csc"
    )
    q = np.zeros(n + m)
    P, q, G, h, A, b, lb, ub = Problem(P, q, G, h, A, b, lb, ub).unpack()
    if G is not None:
        G = spa.hstack([G, spa.csc_matrix((G.shape[0], m))], format="csc")
    if A is not None:
        A = spa.hstack([A, spa.csc_matrix((A.shape[0], m))], format="csc")
    Rx_minus_y = spa.hstack([R, -eye_m], format="csc")
    if A is not None and b is not None:  # help mypy
        A = spa.vstack([A, Rx_minus_y], format="csc")
        b = np.hstack([b, s])
    else:  # no input equality constraint
        A = Rx_minus_y
        b = s
    if lb is not None:
        lb = np.hstack([lb, np.full((m,), -np.inf)])
    if ub is not None:
        ub = np.hstack([ub, np.full((m,), np.inf)])
    xy = solve_qp(
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
        verbose=verbose,
        **kwargs,
    )
    return xy[:n] if xy is not None else None


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
    verbose: bool = False,
    sparse_conversion: Optional[bool] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a constrained weighted linear Least Squares problem.

    The linear least squares is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac12 \| R x - s \|^2_W
                = \frac12 (R x - s)^T W (R x - s) \\
            \mbox{subject to}
                & G x \leq h          \\
                & A x = b             \\
                & lb \leq x \leq ub
        \end{array}\end{split}

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
    verbose :
        Set to `True` to print out extra information.
    sparse_conversion :
        Set to `True` to use a sparse conversion strategy and to `False` to use
        a dense strategy. By default, the conversion strategy to follow is
        determine by the sparsity of :math:`R`. See Notes below.

    Returns
    -------
    :
        Optimal solution if found, otherwise ``None``.

    Note
    ----
    In least squares, the matrix :math:`R` should be symmetric. Many
    solvers (including CVXOPT, OSQP and quadprog) leverage this property and
    may return unintended results when it is not the case. You can set
    project :math:`R` on its symmetric part by:

    .. code:: python

        R = 0.5 * (R + R.transpose())

    Notes
    -----
    This function implements two conversion strategies: one for dense problems
    and one for sparse problems. The conversion strategy to follow can be
    selected via the ``sparse_conversion`` argument. The sparse strategy will
    likely perform better on sparse problems, although this may not always be
    the case (for instance, it may perform worse if :math:`R` has many more
    rows than columns).

    Extra keyword arguments given to this function are forwarded to the
    underlying solvers. For example, OSQP has a setting `eps_abs` which we can
    provide by ``solve_ls(R, s, G, h, solver='osqp', eps_abs=1e-4)``.
    """
    if sparse_conversion is None:
        sparse_conversion = not isinstance(R, np.ndarray)
    if sparse_conversion:
        return __solve_sparse_ls(
            R, s, G, h, A, b, lb, ub, W, solver, initvals, verbose, **kwargs
        )
    return __solve_dense_ls(
        R, s, G, h, A, b, lb, ub, W, solver, initvals, verbose, **kwargs
    )
