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

from numpy import concatenate, eye

from .typing import Matrix, Vector


def convert_bounds_to_inequalities(
    G: Optional[Matrix],
    h: Optional[Vector],
    lb: Optional[Vector],
    ub: Optional[Vector],
) -> Tuple[Optional[Matrix], Optional[Vector]]:
    """
    Append lower or upper bound constraint vectors to inequality constraints.

    G : numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear inequality matrix.
    h : numpy.ndarray, optional
        Linear inequality vector.
    lb: numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Lower bound constraint vector.
    ub: numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Upper bound constraint vector.
    """
    if lb is not None:
        if G is None:
            G = -eye(len(lb))
            h = -lb
        else:  # G is not None and h is not None
            G = concatenate((G, -eye(len(lb))), 0)
            h = concatenate((h, -lb))
    if ub is not None:
        if G is None:
            G = eye(len(ub))
            h = ub
        else:  # G is not None and h is not None
            G = concatenate((G, eye(len(ub))), 0)
            h = concatenate((h, ub))
    return (G, h)
