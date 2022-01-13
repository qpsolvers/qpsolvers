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
Helper function to solve least squares.
"""

from typing import Optional

from numpy import dot, ndarray

from .solve_qp import solve_qp
from .typing import Matrix, Vector


def solve_ls(
    R: Matrix,
    s: Vector,
    G: Optional[Matrix] = None,
    h: Optional[Vector] = None,
    A: Optional[Matrix] = None,
    b: Optional[Vector] = None,
    lb: Optional[Vector] = None,
    ub: Optional[Vector] = None,
    W: Optional[Matrix] = None,
    solver: str = "quadprog",
    initvals: Optional[Vector] = None,
    sym_proj: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Optional[ndarray]:
    """
    Solve a constrained weighted linear Least Squares problem defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\mbox{minimize} &
                \\frac12 \\| R x - s \\|^2_W
                = \\frac12 (R x - s)^T W (R x - s) \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using one of the available QP solvers.

    Parameters
    ----------
    R :
        Symmetric matrix of the cost function (most solvers require it to be
        definite).
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
    lb:
        Lower bound constraint vector.
    ub:
        Upper bound constraint vector.
    W :
        Definite symmetric weight matrix used to define the norm of the cost
        function. The standard L2 norm (W = Identity) is used by default.
    solver :
        Name of the QP solver, to choose in
        :data:`qpsolvers.available_solvers`.
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
    WR: Matrix = R if W is None else dot(W, R)
    P = dot(R.transpose(), WR)
    q = -dot(s.transpose(), WR)
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
