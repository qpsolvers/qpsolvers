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
    G = np.array([[5.0, 2.0, 0.0], [-1.0, 2.0, -1.0]])
    h = np.array([1.0, -2.0])
    A = np.array([1.0, 1.0, 1.0]).reshape((1, 3))
    b = np.array([1.0])
    lb = -0.5 * np.ones(3)
    ub = 1.0 * np.ones(3)
    problem = Problem(P, q, G, h, A, b, lb, ub)

    solution = Solution(problem)
    solution.x = np.array([1.0 / 3, -1.0 / 3, 1.0])
    solution.z = np.array([10.0 / 3, 0.0])
    solution.y = np.array([50.0 / 3])
    solution.z_box = np.array([0.0, 0.0, 37.0 / 3])
    return solution
