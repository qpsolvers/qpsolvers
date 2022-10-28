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

"""Solver interface for SCS"""

from typing import Any, Dict, Optional, Union
from warnings import warn

import numpy as np
import scipy.sparse as spa
from numpy import ndarray
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scs import solve

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
    P: Union[ndarray, csc_matrix],
    q: ndarray,
    G: Optional[Union[ndarray, csc_matrix]] = None,
    h: Optional[ndarray] = None,
    A: Optional[Union[ndarray, csc_matrix]] = None,
    b: Optional[ndarray] = None,
    lb: Optional[ndarray] = None,
    ub: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
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
            & A x = b                    \\\\
            & lb \\leq x \\leq ub
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
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector (not used).
    eps_abs : float
        Absolute feasibility tolerance, see `Termination criteria
        <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`_.
    eps_rel : float
        Relative feasibility tolerance, see `Termination criteria
        <https://www.cvxgrp.org/scs/algorithm/index.html#termination>`_.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded as is to SCS. For instance, you can call
    ``scs_solve_qp(P, q, G, h, normalize=True)``. Solver settings for SCS are
    described `here <https://www.cvxgrp.org/scs/api/settings.html#settings>`_.

    As of SCS 3.0.1, the default feasibility tolerances are set ``1e-4``,
    resulting in larger inequality constraint violations than with other
    solvers on the README and unit test problems. We lower them to ``1e-9`` so
    that SCS behaves closer to the other solvers. If you don't need that much
    precision, increase them for better performance.
    """
    if isinstance(P, ndarray):
        warn_about_sparse_conversion("P")
        P = csc_matrix(P)
    if isinstance(G, ndarray):
        warn_about_sparse_conversion("G")
        G = csc_matrix(G)
    if isinstance(A, ndarray):
        warn_about_sparse_conversion("A")
        A = csc_matrix(A)
    kwargs.update(
        {
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "verbose": verbose,
        }
    )
    data: Dict[str, Any] = {"P": P, "c": q}
    cone: Dict[str, Any] = {}
    if initvals is not None:
        data["x"] = initvals
    if A is not None and b is not None:
        if G is not None and h is not None:
            data["A"] = spa.vstack([A, G], format="csc")
            data["b"] = np.hstack([b, h])
            cone["z"] = b.shape[0]  # zero cone
            cone["l"] = h.shape[0]  # positive cone
        else:  # G is None and h is None
            data["A"] = A
            data["b"] = b
            cone["z"] = b.shape[0]  # zero cone
    elif G is not None and h is not None:
        data["A"] = G
        data["b"] = h
        cone["l"] = h.shape[0]  # positive cone
    else:  # no constraint
        x = lsqr(P, -q)[0]
        if norm(P @ x + q) > 1e-9:
            raise ValueError(
                "problem is unbounded below, "
                "q has component in the nullspace of P"
            )
        return x
    if lb is not None or ub is not None:
        n = P.shape[1]
        cone["bl"] = lb if lb is not None else np.full((n,), -np.inf)
        cone["bu"] = ub if ub is not None else np.full((n,), +np.inf)
        zero_row = csc_matrix((1, n))
        data["A"] = spa.vstack(
            (data["A"], zero_row, -spa.eye(n)),
            format="csc",
        )
        data["b"] = np.hstack((data["b"], 1.0, np.zeros(n)))
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
