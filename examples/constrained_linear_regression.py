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
Test a random QP solver on a constrained linear regression problem.

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
