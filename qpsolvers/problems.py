#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2023 Stéphane Caron and the qpsolvers contributors

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


def get_qpgurdu():
    """Get sample random problem with linear inequality constraints.

    Returns
    -------
    :
        Problem-solution pair.
    """
    P = np.array(
        [
            [
                3.57211988,
                3.04767485,
                2.81378189,
                3.10290601,
                3.70204698,
                3.21624815,
                3.07738552,
                2.97880055,
                2.87282375,
                3.13101137,
            ],
            [
                3.04767485,
                3.29764869,
                2.96655517,
                2.99532101,
                3.27631229,
                2.95993532,
                3.36890754,
                3.41940015,
                2.71055468,
                3.48874903,
            ],
            [
                2.81378189,
                2.96655517,
                4.07209512,
                3.15291684,
                3.25120445,
                3.16570711,
                3.29693401,
                3.57945021,
                2.38634372,
                3.56010605,
            ],
            [
                3.10290601,
                2.99532101,
                3.15291684,
                4.18950328,
                3.80236382,
                3.30578443,
                3.86461151,
                3.73403774,
                2.65103423,
                3.6915013,
            ],
            [
                3.70204698,
                3.27631229,
                3.25120445,
                3.80236382,
                4.49927773,
                3.71882781,
                3.72242148,
                3.36633929,
                3.07400851,
                3.44904275,
            ],
            [
                3.21624815,
                2.95993532,
                3.16570711,
                3.30578443,
                3.71882781,
                3.54009378,
                3.3619341,
                3.45111777,
                2.52760157,
                3.47292034,
            ],
            [
                3.07738552,
                3.36890754,
                3.29693401,
                3.86461151,
                3.72242148,
                3.3619341,
                4.18766506,
                3.9158527,
                2.73687599,
                3.94376429,
            ],
            [
                2.97880055,
                3.41940015,
                3.57945021,
                3.73403774,
                3.36633929,
                3.45111777,
                3.9158527,
                4.4180459,
                2.50596495,
                4.25387869,
            ],
            [
                2.87282375,
                2.71055468,
                2.38634372,
                2.65103423,
                3.07400851,
                2.52760157,
                2.73687599,
                2.50596495,
                2.74656049,
                2.54212279,
            ],
            [
                3.13101137,
                3.48874903,
                3.56010605,
                3.6915013,
                3.44904275,
                3.47292034,
                3.94376429,
                4.25387869,
                2.54212279,
                4.634129,
            ],
        ]
    )
    q = np.array(
        [
            [0.49318579],
            [0.82113304],
            [0.67851692],
            [0.34081485],
            [0.14826526],
            [0.81974126],
            [0.41957706],
            [0.53118637],
            [0.59189664],
            [0.98775649],
        ]
    )
    G = np.array(
        [
            [
                4.38410058e-01,
                4.43204832e-01,
                3.01827071e-01,
                5.77725615e-02,
                8.04962962e-01,
                6.13555163e-01,
                1.15255766e-01,
                7.11331164e-01,
                7.71820534e-02,
                3.86074035e-01,
            ],
            [
                8.47645982e-01,
                9.37475356e-01,
                3.54726656e-01,
                9.64635375e-01,
                5.95008737e-01,
                4.65424573e-01,
                3.60529910e-01,
                5.83149169e-01,
                5.51353698e-01,
                8.45823800e-01,
            ],
            [
                2.29674075e-04,
                5.54870256e-02,
                7.83869376e-01,
                9.97727284e-01,
                1.49512389e-01,
                7.44775614e-01,
                8.76446593e-02,
                2.57348591e-01,
                7.28916655e-01,
                5.97511590e-01,
            ],
            [
                6.92184129e-01,
                9.04600884e-01,
                7.57700115e-01,
                7.76548565e-01,
                5.31039749e-01,
                8.32203998e-01,
                4.27810742e-01,
                1.92236814e-01,
                2.91129478e-01,
                7.76195308e-01,
            ],
            [
                4.73333212e-01,
                3.02129792e-02,
                6.86517354e-01,
                5.08992776e-01,
                8.43205462e-01,
                6.30402967e-01,
                7.92221172e-01,
                3.67768984e-01,
                1.10864990e-01,
                5.44828940e-01,
            ],
            [
                9.23060980e-01,
                4.55743966e-01,
                4.81958856e-02,
                5.47614699e-02,
                8.23194952e-01,
                2.40526659e-01,
                9.33519842e-01,
                5.40430172e-01,
                6.27229337e-01,
                4.27829243e-01,
            ],
            [
                2.39454128e-01,
                1.29688157e-01,
                7.64521599e-01,
                2.66943061e-01,
                4.94990723e-01,
                3.87798160e-01,
                5.76282838e-01,
                8.87340479e-01,
                5.49439650e-01,
                2.99596520e-01,
            ],
            [
                3.73174589e-02,
                4.08407618e-01,
                1.19009418e-01,
                3.02572289e-02,
                1.90287316e-01,
                2.93975786e-01,
                7.65243508e-01,
                8.64670246e-02,
                3.90593097e-01,
                1.33870683e-01,
            ],
            [
                9.10093385e-01,
                9.63382642e-02,
                2.94162739e-01,
                9.71178995e-01,
                1.81811460e-01,
                9.69904715e-02,
                4.10693806e-01,
                7.56873549e-01,
                2.36595007e-01,
                3.19756491e-01,
            ],
            [
                8.58362518e-02,
                7.88161645e-02,
                9.67300428e-01,
                2.59894669e-01,
                1.62774911e-01,
                3.33859109e-01,
                6.15307748e-01,
                1.81164951e-02,
                5.99620503e-01,
                5.71512979e-01,
            ],
        ]
    )
    h = np.array(
        [
            [4.94957567],
            [7.50577326],
            [5.40302286],
            [7.18164978],
            [5.98834884],
            [6.07449251],
            [5.59605532],
            [3.45542914],
            [5.27449417],
            [4.69303926],
        ]
    )
    problem = Problem(P, q, G, h)

    solution = Solution(problem)
    solution.found = True
    return problem, solution


def get_qpgurabs():
    """Get sample random problem with box constraints.

    Returns
    -------
    :
        Problem-solution pair.
    """
    qpgurdu, _ = get_qpgurdu()
    box = np.abs(qpgurdu.h)
    problem = Problem(qpgurdu.P, qpgurdu.q, lb=-box, ub=+box)

    solution = Solution(problem)
    solution.found = True
    return problem, solution


def get_qpgureq():
    """Get sample random problem with equality constraints.

    Returns
    -------
    :
        Problem-solution pair.
    """
    qpgurdu, _ = get_qpgurdu()
    A = qpgurdu.G
    b = 0.1 * qpgurdu.h
    problem = Problem(qpgurdu.P, qpgurdu.q, A=A, b=b)

    solution = Solution(problem)
    solution.found = True
    return problem, solution
