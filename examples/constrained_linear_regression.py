#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test a random QP solver on a constrained linear regression problem.

This example originates from:
    https://stackoverflow.com/a/74422084

See also:
    https://scaron.info/blog/simple-linear-regression-with-online-updates.html
"""

import random

import numpy as np

import qpsolvers
from qpsolvers import solve_ls

a = np.array([1.2, 2.3, 4.2])
b = np.array([1.0, 5.0, 6.0])
c = np.array([5.4, 6.2, 1.9])
m = np.vstack([a, b, c])
y = np.array([5.3, 0.9, 5.6])

# Objective: || [a b c] x - y ||^2
R = m.T
s = y

# Constraint: sum(x) = 1
A = np.ones((1, 3))
b = np.array([1.0])

# Constraint: x >= 0
lb = np.zeros(3)

if __name__ == "__main__":
    solver = random.choice(qpsolvers.available_solvers)
    x = solve_ls(R, s, A=A, b=b, lb=lb, solver=solver)
    print(f"Found solution {x=}")
