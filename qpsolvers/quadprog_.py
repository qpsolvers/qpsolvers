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

"""Solver interface for quadprog"""

from typing import Optional
from warnings import warn

from numpy import hstack, ndarray, vstack
from quadprog import solve_qp


def quadprog_solve_qp(
    P, q, G=None, h=None, A=None, b=None, initvals=None, verbose: bool = False
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

    using `quadprog <https://pypi.python.org/pypi/quadprog/>`_.

    Parameters
    ----------
    P : numpy.array
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic-cost vector.
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
        This argument has no effect, it is here for API conformance.

    Returns
    -------
    x : numpy.array
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    The quadprog solver only considers the lower entries of `P`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    qp_G = P
    qp_a = -q
    if A is not None:
        if G is None:
            qp_C = -A.T
            qp_b = -b  # not None, checked in check_problem_constraints
        else:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T if G is not None else None
        qp_b = -h if h is not None else None
        meq = 0
    try:
        return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
    except ValueError as e:
        error = str(e)
        if "matrix G is not positive definite" in error:
            # quadprog writes G the cost matrix that we write P in this package
            raise ValueError("matrix P is not positive definite") from e
        if "no solution" in error:
            return None
        warn("quadprog raised a ValueError: {}".format(e))
        return None
