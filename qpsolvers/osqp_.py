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

"""Solver interface for OSQP"""

from typing import Optional
from warnings import warn

import osqp

from numpy import hstack, inf, ndarray, ones
from osqp import OSQP
from scipy.sparse import csc_matrix, vstack


def conversion_warning(M):
    """Return conversion warning message for a given matrix name."""
    return (
        "Converted %s to scipy.sparse.csc.csc_matrix\n"
        "For best performance, build %s as a csc_matrix "
        "rather than as a numpy.ndarray" % (M, M)
    )


def osqp_solve_qp(
    P,
    q,
    G=None,
    h=None,
    A=None,
    b=None,
    initvals=None,
    verbose: bool = False,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    polish: bool = True,
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

    using `OSQP <https://github.com/oxfordcontrol/osqp>`_.

    Parameters
    ----------
    P : scipy.sparse.csc_matrix
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic cost vector.
    G : scipy.sparse.csc_matrix
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector.
    verbose : bool, optional
        Set to `True` to print out extra information.
    eps_abs : float, optional
        Absolute convergence tolerance of the solver. Lower values yield more
        precise solutions at the cost of computation time.
    eps_rel : float, optional
        Relative convergence tolerance of the solver. Lower values yield more
        precise solutions at the cost of computation time.
    polish : bool, optional
        Perform `polishing <https://osqp.org/docs/solver/#polishing>`_, an
        additional step where the solver tries to improve the accuracy of the
        solution. Default is ``True``.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.

    Note
    ----
    As of OSQP v0.6.1, the default values for both absolute and relative
    tolerances are set to ``1e-3``, which results in low solver times but
    imprecise solutions compared to the other QP solvers. We lower them to
    ``1e-5`` so that OSQP behaves closer to the norm in terms of numerical
    accuracy.
    """
    if isinstance(P, ndarray):
        warn(conversion_warning("P"))
        P = csc_matrix(P)
    solver = OSQP()
    kwargs = {
        "eps_abs": eps_abs,
        "eps_rel": eps_rel,
        "polish": polish,
        "verbose": verbose,
    }
    if A is None and G is None:
        solver.setup(P=P, q=q, **kwargs)
    elif A is not None:
        if isinstance(A, ndarray):
            warn(conversion_warning("A"))
            A = csc_matrix(A)
        if G is None:
            solver.setup(P=P, q=q, A=A, l=b, u=b, **kwargs)
        else:  # G is not None
            l_inf = -inf * ones(len(h))
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l_inf, b])
            qp_u = hstack([h, b])
            solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, **kwargs)
    else:  # A is None
        if isinstance(G, ndarray):
            warn(conversion_warning("G"))
            G = csc_matrix(G)
        l_inf = -inf * ones(len(h))
        solver.setup(P=P, q=q, A=G, l=l_inf, u=h, **kwargs)
    if initvals is not None:
        solver.warm_start(x=initvals)
    res = solver.solve()
    if hasattr(solver, "constant"):
        success_status = solver.constant("OSQP_SOLVED")
    else:  # more recent versions of OSQP
        success_status = osqp.constant("OSQP_SOLVED")
    if res.info.status_val != success_status:
        warn("OSQP exited with status '%s'" % res.info.status)
        return None
    return res.x
