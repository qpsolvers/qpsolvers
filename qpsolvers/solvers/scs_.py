#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stéphane Caron <stephane.caron@normalesup.org>
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

from .typing import DenseOrCSCMatrix
from .typing import warn_about_sparse_conversion


# See https://www.cvxgrp.org/scs/api/exit_flags.html#exit-flags
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
    P: DenseOrCSCMatrix,
    q: ndarray,
    G: Optional[DenseOrCSCMatrix] = None,
    h: Optional[ndarray] = None,
    A: Optional[DenseOrCSCMatrix] = None,
    b: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
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

    Notes
    -----
    Keyword arguments are forwarded as is to SCS. For instance, you can call
    ``scs_solve_qp(P, q, G, h,normalize=True)``. Solver settings for SCS are
    described `here <https://www.cvxgrp.org/scs/api/settings.html#settings>`_.
    """
    if isinstance(P, ndarray):
        warn_about_sparse_conversion("P")
        P = sparse.csc_matrix(P)
    if isinstance(G, ndarray):
        warn_about_sparse_conversion("G")
        G = sparse.csc_matrix(G)
    if isinstance(A, ndarray):
        warn_about_sparse_conversion("A")
        A = sparse.csc_matrix(A)
    kwargs["verbose"] = verbose
    data = {"P": P, "c": q}
    cone = {}
    if initvals is not None:
        data["x"] = initvals
    if G is None or h is None:
        raise ValueError("SCS cannot work with unconstrained problems")
    elif A is not None and b is not None:
        data["A"] = sparse.vstack([A, G], format="csc")
        data["b"] = hstack([b, h])
        cone["z"] = b.shape[0]  # zero cone
        cone["l"] = h.shape[0]  # positive orthant
    else:  # G is not None and h is not None
        data["A"] = G
        data["b"] = h
        cone["l"] = h.shape[0]  # positive orthant
    solution = solve(data, cone, **kwargs)
    status_val = solution["info"]["status_val"]
    if status_val != 1:
        warn(
            f"SCS returned {status_val}: {__status_val_meaning__[status_val]}"
        )
        if status_val != 2:
            # solution is not inaccurate, so the optimization failed
            return None
    return solution["x"]
