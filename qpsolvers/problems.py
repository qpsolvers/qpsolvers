#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2023 the qpsolvers contributors.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Collection of sample problems."""

from typing import Tuple

import numpy as np
import scipy.sparse as spa

from .problem import Problem
from .solution import Solution


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


def get_qpsut01() -> Tuple[Problem, Solution]:
    """Get QPSUT01 problem and its solution.

    Returns
    -------
    :
        Problem-solution pair.
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
    solution.found = True
    solution.x = np.array([0.4, -0.4, 1.0])
    solution.z = np.array([0.0, 0.0])
    solution.y = np.array([-5.8])
    solution.z_box = np.array([0.0, -1.8, 3.0])
    return problem, solution


def get_qpsut02() -> Tuple[Problem, Solution]:
    """Get QPSUT02 problem and its solution.

    Returns
    -------
    :
        Problem-solution pair.
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
    solution.found = True
    solution.x = np.array([1.36597938, -1.0, 6.0, 3.63402062])
    solution.z = np.array([0.0])
    solution.y = np.array([-377.60314303, -62.75251185])  # YMMV
    solution.z_box = np.array([0.0, -138.9585918, 37.53106937, 0.0])  # YMMV
    return problem, solution


def get_qpsut03() -> Tuple[Problem, Solution]:
    """Get QPSUT03 problem and its solution.

    Returns
    -------
    :
        Problem-solution pair.

    Notes
    -----
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
    solution.found = True
    solution.x = np.array([0.18143455, 0.00843864, -2.35442995, 0.35443034])
    solution.z = np.array([])
    solution.y = np.array([])
    solution.z_box = np.array([0.0, 0.0, 0.0, 0.0])
    return problem, solution


def get_qpsut04() -> Tuple[Problem, Solution]:
    """Get QPSUT04 problem and its solution.

    Returns
    -------
    :
        Problem-solution pair.
    """
    n = 3
    P = np.eye(n)
    q = 0.01 * np.ones(shape=(n, 1))  # non-flat vector
    G = np.eye(n)
    h = np.ones(shape=(n,))
    A = np.ones(shape=(n,))
    b = np.ones(shape=(1,))
    problem = Problem(P, q, G, h, A, b)

    solution = Solution(problem)
    solution.found = True
    solution.x = 1.0 / 3.0 * np.ones(n)
    solution.y = np.array([1.0 / 3.0 + 0.01])
    solution.z = np.zeros(n)
    return problem, solution


def get_qpsut05() -> Tuple[Problem, Solution]:
    """Get QPSUT05 problem and its solution.

    Returns
    -------
    :
        Problem-solution pair.
    """
    P = np.array([2.0])
    q = np.array([-2.0])
    problem = Problem(P, q)

    solution = Solution(problem)
    solution.found = True
    solution.x = np.array([1.0])
    return problem, solution


def get_qptest():
    """Get QPTEST problem from the Maros-Meszaros test set.

    Returns
    -------
    :
        Problem-solution pair.
    """
    P = np.array([[8.0, 2.0], [2.0, 10.0]])
    q = np.array([1.5, -2.0])
    G = np.array([[-1.0, 2.0], [-2.0, -1.0]])
    h = np.array([6.0, -2.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([20.0, np.inf])
    problem = Problem(P, q, G, h, lb=lb, ub=ub)

    solution = Solution(problem)
    solution.found = True
    solution.x = np.array([0.7625, 0.475])
    solution.z = np.array([0.0, 4.275])
    solution.z_box = np.array([0.0, 0.0])
    return problem, solution
