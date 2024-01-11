#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

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
