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

from qpsolvers import solve_ls, print_matrix_vector
from time import perf_counter

R = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
s = np.array([3.0, 2.0, 3.0])
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0]).reshape((3,))

start_time = perf_counter()
solver = "quadprog"  # see qpsolvers.available_solvers
x = solve_ls(R, s, G, h, solver=solver, verbose=True)
end_time = perf_counter()

print("")
print("    min. || R * x - s ||^2")
print("    s.t. G * x <= h")
print("")
print_matrix_vector(R, "R", s, "s")
print("")
print_matrix_vector(G, "G", h, "h")
print("")
print(f"Solution: x = {x}")
print(f"Solve time: {1e6 * (end_time - start_time):.0f} [us]")
print(f"Solver: {solver}")
