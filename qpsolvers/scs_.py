#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2020 Stephane Caron <stephane.caron@normalesup.org>
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

from numpy import hstack, ndarray
from scs import solve
from scipy import sparse

from .socp import convert_to_socp


# See https://github.com/cvxgrp/scs/blob/master/include/glbopts.h
__status_val_meaning__ = {
    -7: "INFEASIBLE_INACCURATE",
    -6: "UNBOUNDED_INACCURATE",
    -5: "SIGINT",
    -4: "FAILED",
    -3: "INDETERMINATE",
    -2: "INFEASIBLE (primal infeasible, dual unbounded)",
    -1: "UNBOUNDED (primal unbounded, dual infeasible)",
    0: "UNFINISHED (never returned, used as placeholder)",
    1: "SOLVED",
    2: "SOLVED_INACCURATE",
}


def scs_solve_qp(
    P, q, G=None, h=None, A=None, b=None, initvals=None, verbose: bool = False, **kwargs
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = h
        \\end{array}\\end{split}

    using `SCS <https://github.com/cvxgrp/scs>`_.

    Parameters
    ----------
    P : numpy.array
        Primal quadratic cost matrix.
    q : numpy.array
        Primal quadratic cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).
    verbose : bool, optional
        Set to `True` to print out extra information.

    Note
    ----
    All other keyword arguments are forwarded to the SCS solver. For instance, you can
    call ``scs_solve_qp(P, q, G, h, use_indirect=True, normalize=True)``. See the solver
    documentation for details.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    c_socp, G_socp, h_socp, dims = convert_to_socp(P, q, G, h)
    if A is not None:
        dims["f"] = A.shape[0]  # number of equality constraints
        A_socp = sparse.hstack([A, sparse.csc_matrix((A.shape[0], 1))], format="csc")
        A_scs = sparse.vstack([A_socp, G_socp], format="csc")
        b_scs = hstack([b, h_socp])
        data = {"A": A_scs, "b": b_scs, "c": c_socp}
        solution = solve(data, dims, verbose=verbose, **kwargs)
    else:
        data = {"A": G_socp, "b": h_socp, "c": c_socp}
        solution = solve(data, dims, verbose=verbose, **kwargs)
    status_val = solution["info"]["statusVal"]
    if status_val != 1:
        warn(f"SCS returned {status_val}: {__status_val_meaning__[status_val]}")
        if status_val != 2:
            # it's not that the solution is inaccurate, so the optimization failed
            return None
    return solution["x"][:-1]
