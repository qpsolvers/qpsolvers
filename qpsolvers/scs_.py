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

"""Solver interface for SCS"""

from typing import Optional
from warnings import warn

from numpy import hstack, ndarray
from scs import solve
from scipy import sparse

from .warnings import warn_about_conversion


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
    P,
    q,
    G=None,
    h=None,
    A=None,
    b=None,
    initvals=None,
    verbose: bool = False,
    use_indirect: bool = True,
    **kwargs,
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
    use_indirect: bool, optional
        Solve linear systems either "directly" via a sparse LDL factorization or
        "indirectly" by means of a `conjugate gradient method
        <https://stanford.edu/~boyd/papers/pdf/scs_long.pdf>`_.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    All other keyword arguments are forwarded as is to SCS. For instance, you can call
    ``scs_solve_qp(P, q, G, h, use_indirect=True, normalize=True)``. See the solver
    documentation for details.
    """
    if isinstance(P, ndarray):
        warn_about_conversion("P")
        P = sparse.csc_matrix(P)
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    data = {"P": P, "c": q}
    dims = {"l": G.shape[0]}
    kwargs.update({"use_indirect": use_indirect, "verbose": verbose})
    if A is not None:
        dims["f"] = A.shape[0]  # number of equality constraints
        data["A"] = sparse.vstack([A, G], format="csc")
        data["b"] = hstack([b, h])
    else:  # only inequality constraints
        if isinstance(G, ndarray):
            warn_about_conversion("G")
            G = sparse.csc_matrix(G)
        data["A"] = G
        data["b"] = h
    solution = solve(data, dims, **kwargs)
    status_val = solution["info"]["statusVal"]
    if status_val != 1:
        warn(f"SCS returned {status_val}: {__status_val_meaning__[status_val]}")
        if status_val != 2:
            # it's not that the solution is inaccurate, so the optimization failed
            return None
    return solution["x"]
