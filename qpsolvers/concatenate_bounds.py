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

"""Functions to convert vector bounds into linear inequality constraints."""

from typing import Optional, Tuple

import numpy as np

from numpy import concatenate, ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from .typing import Matrix, Vector


try:
    import cvxopt

    def cvxopt_concatenate(G, sign: float, n: int):
        """
        Concatenate the sparse matrix `sign * eye(n)` to a CVXOPT matrix `G`.

        Parameters
        ----------
        G :
            Linear inequality matrix.
        sign :
            Sign factor: -1.0 for a lower and +1.0 for an upper bound.
        n :
            Dimension of the identity matrix, which in context should also be
            the number of optimization variables.

        Returns
        -------
        G :
            Updated linear inequality matrix.
        """
        return cvxopt.sparse(
            [
                G,
                sign * cvxopt.spmatrix(1.0, range(n), range(n)),
            ]
        )

except ImportError:

    def cvxopt_concatenate(G, sign: float, n: int):
        """
        This function is not available because CVXOPT is not installed.
        """
        raise TypeError(
            "Inequality matrix G has type cvxopt.spmatrix "
            "(it is neither an ndarray nor a csc_matrix), "
            "but CVXOPT is not installed"
        )


def concatenate_bound(
    G: Optional[Matrix],
    h: Optional[Vector],
    b: Vector,
    sign: float,
) -> Tuple[Optional[Matrix], Optional[Vector]]:
    """
    Append bound constraint vectors to inequality constraints.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    b:
        Bound constraint vector.
    sign:
        Sign factor: -1.0 for a lower and +1.0 for an upper bound.

    Returns
    -------
    G : numpy.ndarray, scipy.sparse.csc_matrix, cvxopt.spmatrix, or None
        Updated linear inequality matrix.
    h : numpy.ndarray or None
        Updated linear inequality vector.
    """
    n = len(b)  # == number of optimization variables
    if G is None or h is None:
        G = sign * np.eye(n)
        h = sign * b
    else:  # G is not None and h is not None
        if isinstance(G, ndarray):
            G = concatenate((G, sign * np.eye(n)), 0)
        elif isinstance(G, csc_matrix):
            G = sparse.vstack([G, sign * sparse.eye(n)], format="csc")
        else:  # isinstance(G, cvxopt.spmatrix)
            G = cvxopt_concatenate(G, sign, n)
        h = concatenate((h, sign * b))
    return (G, h)


def concatenate_bounds(
    G: Optional[Matrix],
    h: Optional[Vector],
    lb: Optional[Vector],
    ub: Optional[Vector],
) -> Tuple[Optional[Matrix], Optional[Vector]]:
    """
    Append lower or upper bound constraint vectors to inequality constraints.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    lb:
        Lower bound constraint vector.
    ub:
        Upper bound constraint vector.

    Returns
    -------
    G : numpy.ndarray, scipy.sparse.csc_matrix, cvxopt.spmatrix, or None
        Updated linear inequality matrix.
    h : numpy.ndarray or None
        Updated linear inequality vector.
    """
    if lb is not None:
        G, h = concatenate_bound(G, h, lb, -1.0)
    if ub is not None:
        G, h = concatenate_bound(G, h, ub, +1.0)
    return (G, h)
