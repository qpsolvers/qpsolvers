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

import numpy as np

from qpsolvers import Problem


def get_sd3310_problem() -> Problem:
    """
    Get a small dense problem with 3 optimization variables, 3 inequality
    constraints, 1 equality constraint and 0 box constraint.
    """
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
    G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
    h = np.array([3.0, 2.0, -2.0]).reshape((3,))
    A = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0])
    return Problem(P, q, G, h, A, b)


def get_qpmad_demo_problem():
    """
    Problem from qpmad's `demo.cpp
    <https://github.com/asherikov/qpmad/blob/5e4038f15d85a2a396bb062599f9d7a06d0b0764/test/dependency/demo.cpp>`__.
    """
    P = np.eye(20)
    q = np.ones((20,))
    G = np.vstack([np.ones((1, 20)), -np.ones((1, 20))])
    h = np.hstack([1.5, 1.5])
    lb = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ]
    )
    ub = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    return Problem(P, q, G, h, lb=lb, ub=ub)
