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

from .convert_to_socp import convert_to_socp


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
    P: ndarray,
    q: ndarray,
    G: Optional[ndarray] = None,
    h: Optional[ndarray] = None,
    A: Optional[ndarray] = None,
    b: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
    eps: float = 1e-7,
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
    eps :
        Convergence tolerange.
    use_indirect:
        Solve linear systems either "directly" via a sparse LDL factorization
        or "indirectly" by means of a `conjugate gradient method
        <https://stanford.edu/~boyd/papers/pdf/scs_long.pdf>`_.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    As of SCS 2.1.2, the default convergence tolerance ``eps`` is set to
    ``1e-5``, resulting in inequality constraints that are violated by more
    than ``1e-6`` as opposed to ``1e-10`` for other solvers e.g. on the README
    problem. We lower it to ``1e-7`` and switch ``use_indirect=True`` so that
    SCS behaves closer to the other solvers on this example.

    All other keyword arguments are forwarded as is to SCS. For instance, you
    can call ``scs_solve_qp(P, q, G, h, use_indirect=True, normalize=True)``.
    See the solver documentation for details.
    """
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    c_socp, G_socp, h_socp, dims = convert_to_socp(P, q, G, h)
    kwargs.update({"eps": eps, "use_indirect": use_indirect, "verbose": verbose})
    if A is not None and b is not None:
        dims["f"] = A.shape[0]  # number of equality constraints
        A_socp = sparse.hstack([A, sparse.csc_matrix((A.shape[0], 1))], format="csc")
        A_scs = sparse.vstack([A_socp, G_socp], format="csc")
        b_scs = hstack([b, h_socp])
        data = {"A": A_scs, "b": b_scs, "c": c_socp}
        solution = solve(data, dims, **kwargs)
    else:
        data = {"A": G_socp, "b": h_socp, "c": c_socp}
        solution = solve(data, dims, **kwargs)
    status_val = solution["info"]["statusVal"]
    if status_val != 1:
        warn(f"SCS returned {status_val}: {__status_val_meaning__[status_val]}")
        if status_val != 2:
            # solution is not inaccurate, so the optimization failed
            return None
    return solution["x"][:-1]
