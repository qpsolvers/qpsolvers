#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Apply lasso regularization to a quadratic program.

Details:
    https://scaron.info/blog/lasso-regularization-in-quadratic-programming.html
"""

import numpy as np

from qpsolvers import solve_qp

# Objective: || R x - s ||^2
n = 6
R = np.diag(range(1, n + 1))
s = np.ones(n)

# Convert our least-squares objective to quadratic programming
P = np.dot(R.transpose(), R)
q = -np.dot(s.transpose(), R)

# Linear inequality constraints: G x <= h
G = np.array(
    [
        [1.0, 0.0] * (n // 2),
        [0.0, 1.0] * (n // 2),
    ]
)
h = np.array([10.0, -10.0])

# Lasso parameter
t: float = 10.0

# Lasso: inequality constraints
G_lasso = np.vstack(
    [
        np.hstack([G, np.zeros((G.shape[0], n))]),
        np.hstack([+np.eye(n), -np.eye(n)]),
        np.hstack([-np.eye(n), -np.eye(n)]),
        np.hstack([np.zeros((1, n)), np.ones((1, n))]),
    ]
)
h_lasso = np.hstack([h, np.zeros(n), np.zeros(n), t])

# Lasso: objective
P_lasso = np.vstack(
    [
        np.hstack([P, np.zeros((n, n))]),
        np.zeros((n, 2 * n)),
    ]
)
q_lasso = np.hstack([q, np.ones(n)])

if __name__ == "__main__":
    x_unreg = solve_qp(P, q, G, h, solver="proxqp")
    print(f"Solution without lasso: {x_unreg = }")

    lasso_res = solve_qp(P_lasso, q_lasso, G_lasso, h_lasso, solver="proxqp")
    x_lasso = lasso_res[:n]
    z_lasso = lasso_res[n:]
    print(f"Solution with lasso ({t=}): {x_lasso = }")
    print(f"We can check that abs(x_lasso) = {z_lasso = }")
