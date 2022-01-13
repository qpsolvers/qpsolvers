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

"""Solver interface for ECOS"""

from typing import Optional
from warnings import warn

from ecos import solve
from numpy import ndarray
from scipy import sparse

from .convert_to_socp import convert_to_socp


__exit_flag_meaning__ = {
    0: "OPTIMAL",
    1: "PRIMAL INFEASIBLE",
    2: "DUAL INFEASIBLE",
    -1: "MAXIT REACHED",
}


def ecos_solve_qp(
    P: ndarray,
    q: ndarray,
    G: Optional[ndarray] = None,
    h: Optional[ndarray] = None,
    A: Optional[ndarray] = None,
    b: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b
        \\end{array}\\end{split}

    using `ECOS <https://github.com/embotech/ecos>`_.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    c_socp, G_socp, h_socp, dims = convert_to_socp(P, q, G, h)
    if A is not None:
        A_socp = sparse.hstack(
            [A, sparse.csc_matrix((A.shape[0], 1))], format="csc"
        )
        solution = solve(
            c_socp, G_socp, h_socp, dims, A_socp, b, verbose=verbose
        )
    else:
        solution = solve(c_socp, G_socp, h_socp, dims, verbose=verbose)
    flag = solution["info"]["exitFlag"]
    if flag != 0:
        warn(f"ECOS returned exit flag {flag} ({__exit_flag_meaning__[flag]})")
        return None
    return solution["x"][:-1]
