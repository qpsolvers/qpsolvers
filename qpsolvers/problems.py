#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2023 the qpsolvers contributors.
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

"""Collection of sample problems."""

import numpy as np
import scipy.sparse as spa


def get_sparse_least_squares(n):
    """
    Get a sparse least squares problem.

    Parameters
    ----------
    n :
        Problem size.

    Notes
    -----
    This problem was inspired by `this question on Stack Overflow
    <https://stackoverflow.com/q/73656257/3721564>`__.
    """
    # minimize 1/2 || x - s ||^2
    R = spa.eye(n, format="csc")
    s = np.array(range(n), dtype=float)

    # such that G * x <= h
    G = spa.diags(
        diagonals=[
            [1.0 if i % 2 == 0 else 0.0 for i in range(n)],
            [1.0 if i % 3 == 0 else 0.0 for i in range(n - 1)],
            [1.0 if i % 5 == 0 else 0.0 for i in range(n - 1)],
        ],
        offsets=[0, 1, -1],
        format="csc",
    )
    h = np.ones(G.shape[0])

    # such that sum(x) == 42
    A = spa.csc_matrix(np.ones((1, n)))
    b = np.array([42.0]).reshape((1,))

    # such that x >= 0
    lb = np.zeros(n)
    ub = None

    return R, s, G, h, A, b, lb, ub
