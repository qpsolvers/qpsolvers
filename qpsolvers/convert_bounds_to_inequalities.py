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

from typing import Optional, Tuple

import cvxopt
import numpy as np

from numpy import concatenate, ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from .typing import Matrix, Vector


def convert_bounds_to_inequalities(
    G: Optional[Matrix],
    h: Optional[Vector],
    lb: Optional[Vector],
    ub: Optional[Vector],
) -> Tuple[Optional[Matrix], Optional[Vector]]:
    """
    Append lower or upper bound constraint vectors to inequality constraints.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear inequality matrix.
    h : numpy.ndarray, optional
        Linear inequality vector.
    lb: numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Lower bound constraint vector.
    ub: numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Upper bound constraint vector.

    Returns
    -------
    G : numpy.ndarray, scipy.sparse.csc_matrix, cvxopt.spmatrix, or None
        Linear inequality matrix.
    h : numpy.ndarray or None
        Linear inequality vector.
    """
    if lb is not None:
        if G is None:
            G = -np.eye(len(lb))
            h = -lb
        else:  # G is not None and h is not None
            if isinstance(G, ndarray):
                G = concatenate((G, -np.eye(len(lb))), 0)
            elif isinstance(G, csc_matrix):
                G = sparse.vstack([G, -sparse.eye(len(lb))])
            else:  # isinstance(G, cvxopt.spmatrix)
                G = cvxopt.sparse(
                    [
                        [G],
                        -cvxopt.spmatrix(1.0, range(len(lb)), range(len(lb))),
                    ]
                )
            h = concatenate((h, -lb))
    if ub is not None:
        if G is None:
            G = np.eye(len(ub))
            h = ub
        else:  # G is not None and h is not None
            if isinstance(G, ndarray):
                G = concatenate((G, np.eye(len(ub))), 0)
            elif isinstance(G, csc_matrix):
                G = sparse.vstack([G, sparse.eye(len(ub))])
            else:  # isinstance(G, cvxopt.spmatrix)
                G = cvxopt.sparse(
                    [
                        [G],
                        cvxopt.spmatrix(1.0, range(len(ub)), range(len(ub))),
                    ]
                )
            h = concatenate((h, ub))
    return (G, h)
