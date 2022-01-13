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
Test the "quadprog" QP solver on a small dense problem.
"""

import numpy as np

from qpsolvers import solve_qp, print_matrix_vector
from time import perf_counter

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = np.dot(M.T, M)  # this is a positive definite matrix
q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0]).reshape((3,))
A = np.array([1.0, 1.0, 1.0])
b = np.array([1.0])

start_time = perf_counter()
solver = "quadprog"  # see qpsolvers.available_solvers
x = solve_qp(P, q, G, h, A, b, solver=solver)
end_time = perf_counter()

print("")
print("    min. 1/2 x^T P x + q^T x")
print("    s.t. G * x <= h")
print("         A * x == b")
print("")
print_matrix_vector(P, "P", q, "q")
print("")
print_matrix_vector(G, "G", h, "h")
print("")
print_matrix_vector(A, "A", b, "b")
print("")
print(f"Solution: x = {x}")
print(f"Solve time: {1e6 * (end_time - start_time):.0f} [us]")
print(f"Solver: {solver}")
