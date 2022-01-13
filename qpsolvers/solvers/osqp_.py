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

"""Solver interface for OSQP"""

from typing import Optional
from warnings import warn

import osqp

from numpy import hstack, inf, ndarray, ones
from osqp import OSQP
from scipy import sparse

from .typing import DenseOrCSCMatrix
from .typing import warn_about_sparse_conversion


def osqp_solve_qp(
    P: DenseOrCSCMatrix,
    q: DenseOrCSCMatrix,
    G: Optional[DenseOrCSCMatrix] = None,
    h: Optional[DenseOrCSCMatrix] = None,
    A: Optional[DenseOrCSCMatrix] = None,
    b: Optional[DenseOrCSCMatrix] = None,
    initvals: Optional[DenseOrCSCMatrix] = None,
    verbose: bool = False,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    polish: bool = True,
    **kwargs
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

    using `OSQP <https://github.com/oxfordcontrol/osqp>`_.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix.
    q :
        Quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.
    eps_abs :
        Absolute convergence tolerance of the solver. Lower values yield more
        precise solutions at the cost of computation time.
    eps_rel :
        Relative convergence tolerance of the solver. Lower values yield more
        precise solutions at the cost of computation time.
    polish :
        Perform `polishing <https://osqp.org/docs/solver/#polishing>`_, an
        additional step where the solver tries to improve the accuracy of the
        solution. Default is ``True``.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    OSQP requires a symmetric `P` and won't check for errors otherwise. Check
    out this point if you `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.

    Notes
    -----
    As of OSQP 0.6.1, the default values for both absolute and relative
    tolerances are set to ``1e-3``, which results in low solver times but
    imprecise solutions compared to the other QP solvers. We lower them to
    ``1e-5`` so that OSQP behaves closer to the other solvers in terms of
    numerical accuracy.

    All other keyword arguments are forwarded to the OSQP solver. For instance,
    you can call ``osqp_solve_qp(P, q, G, h, u, alpha=1.42)``. See the `solver
    documentation <https://osqp.org/docs/interfaces/solver_settings.html>`_ for
    details.
    """
    if isinstance(P, ndarray):
        warn_about_sparse_conversion("P")
        P = sparse.csc_matrix(P)
    solver = OSQP()
    kwargs = {
        "eps_abs": eps_abs,
        "eps_rel": eps_rel,
        "polish": polish,
        "verbose": verbose,
    }
    if A is not None and b is not None:
        if isinstance(A, ndarray):
            warn_about_sparse_conversion("A")
            A = sparse.csc_matrix(A)
        if G is not None and h is not None:
            l_inf = -inf * ones(len(h))
            qp_A = sparse.vstack([G, A], format="csc")
            qp_l = hstack([l_inf, b])
            qp_u = hstack([h, b])
            solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, **kwargs)
        else:  # no inequality constraint
            solver.setup(P=P, q=q, A=A, l=b, u=b, **kwargs)
    elif G is not None and h is not None:
        if isinstance(G, ndarray):
            warn_about_sparse_conversion("G")
            G = sparse.csc_matrix(G)
        l_inf = -inf * ones(len(h))
        solver.setup(P=P, q=q, A=G, l=l_inf, u=h, **kwargs)
    else:  # no inequality nor equality constraint
        solver.setup(P=P, q=q, **kwargs)
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
