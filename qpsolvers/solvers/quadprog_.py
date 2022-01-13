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

"""Solver interface for quadprog"""

from typing import Optional
from warnings import warn

from numpy import hstack, ndarray, vstack
from quadprog import solve_qp


def quadprog_solve_qp(
    P: ndarray,
    q: ndarray,
    G: Optional[ndarray] = None,
    h: Optional[ndarray] = None,
    A: Optional[ndarray] = None,
    b: Optional[ndarray] = None,
    initvals: ndarray = None,
    verbose: bool = False,
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

    using `quadprog <https://pypi.python.org/pypi/quadprog/>`_.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix.
    q :
        Quadratic-cost vector.
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

    Note
    ----
    The quadprog solver only considers the lower entries of :math:`P`,
    therefore it will use a different cost than the one intended if a
    non-symmetric matrix is provided.

    Notes
    -----
    All other keyword arguments are forwarded to the quadprog solver. For
    instance, you can call ``quadprog_solve_qp(P, q, G, h, factorized=True)``.
    See the solver documentation for details.
    """
    if initvals is not None and verbose:
        warn("note that warm-start values ignored by quadprog")
    qp_G = P
    qp_a = -q
    qp_C: Optional[ndarray] = None
    qp_b: Optional[ndarray] = None
    if A is not None and b is not None:
        if G is not None and h is not None:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
        else:
            qp_C = -A.T
            qp_b = -b
        meq = A.shape[0]
    else:  # no equality constraint
        if G is not None and h is not None:
            qp_C = -G.T
            qp_b = -h
        meq = 0
    try:
        return solve_qp(qp_G, qp_a, qp_C, qp_b, meq, **kwargs)[0]
    except ValueError as e:
        error = str(e)
        if "matrix G is not positive definite" in error:
            # quadprog writes G the cost matrix that we write P in this package
            raise ValueError("matrix P is not positive definite") from e
        if "no solution" in error:
            return None
        warn("quadprog raised a ValueError: {}".format(e))
        return None
