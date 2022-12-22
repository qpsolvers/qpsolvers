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

"""
Reference problems with their solutions.
"""

import numpy as np

from qpsolvers import Problem, Solution


def get_qpsut01() -> Solution:
    """
    Get QPSUT01 problem and its solution.
    """
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([3.0, 2.0, 3.0]), M)
    G = np.array([[4.0, 2.0, 0.0], [-1.0, 2.0, -1.0]])
    h = np.array([1.0, -2.0])
    A = np.array([1.0, 1.0, 1.0]).reshape((1, 3))
    b = np.array([1.0])
    lb = np.array([-0.5, -0.4, -0.5])
    ub = np.array([1.0, 1.0, 1.0])
    problem = Problem(P, q, G, h, A, b, lb, ub)

    solution = Solution(problem)
    solution.x = np.array([0.4, -0.4, 1.0])
    solution.z = np.array([0.0, 0.0])
    solution.y = np.array([-5.8])
    solution.z_box = np.array([0.0, -1.8, 3.0])
    return solution


def get_qpsut02() -> Solution:
    """
    Get QPSUT02 problem and its solution.
    """
    M = np.array(
        [
            [1.0, -2.0, 0.0, 8.0],
            [-6.0, 3.0, 1.0, 4.0],
            [-2.0, 1.0, 0.0, 1.0],
            [9.0, 9.0, 5.0, 3.0],
        ]
    )
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([-3.0, 2.0, 0.0, 9.0]), M)
    G = np.array(
        [
            [4.0, 7.0, 0.0, -2.0],
        ]
    )
    h = np.array([30.0])
    A = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    b = np.array([10.0, 0.0])
    lb = np.array([-2.0, -1.0, -3.0, 1.0])
    ub = np.array([4.0, 2.0, 6.0, 10.0])
    problem = Problem(P, q, G, h, A, b, lb, ub)

    solution = Solution(problem)
    solution.x = np.array([1.36597938, -1.0, 6.0, 3.63402062])
    solution.z = np.array([0.0])
    solution.y = np.array([-377.60314303, -62.75251185])  # YMMV
    solution.z_box = np.array([0.0, -138.9585918, 37.53106937, 0.0])  # YMMV
    return solution


def get_qpsut03() -> Solution:
    """
    Get QPSUT03 problem and its solution.

    Notes:
        This problem has partial box bounds, that is, -infinity on some lower
        bounds and +infinity on some upper bounds.
    """
    M = np.array(
        [
            [1.0, -2.0, 0.0, 8.0],
            [-6.0, 3.0, 1.0, 4.0],
            [-2.0, 1.0, 0.0, 1.0],
            [9.0, 9.0, 5.0, 3.0],
        ]
    )
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([-3.0, 2.0, 0.0, 9.0]), M)
    G = None
    h = None
    A = None
    b = None
    lb = np.array([-np.inf, -0.4, -np.inf, -1.0])
    ub = np.array([np.inf, np.inf, 0.5, 1.0])
    problem = Problem(P, q, G, h, A, b, lb, ub)

    solution = Solution(problem)
    solution.x = np.array([0.4, -0.4, 1.0])
    solution.z = np.array([0.0, 0.0])
    solution.y = np.array([-5.8])
    solution.z_box = np.array([0.0, -1.8, 3.0])
    return solution
