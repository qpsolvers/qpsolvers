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
Get both primal and dual solutions to a quadratic program.
"""

import argparse
import random
from time import perf_counter

import numpy as np

from qpsolvers import available_solvers, print_matrix_vector, solve_qp2

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = np.dot(M.T, M)  # this is a positive definite matrix
q = np.dot(np.array([3.0, 2.0, 3.0]), M)
G = np.array([[5.0, 2.0, 0.0], [-1.0, 2.0, -1.0]])
h = np.array([1.0, -2.0])
A = np.array([1.0, 1.0, 1.0]).reshape((1, 3))
b = np.array([1.0])
lb = -0.5 * np.ones(3)
ub = 1.0 * np.ones(3)

if False:
    G[0, 0] = 4.0
    lb[1] = -0.4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("solver")
    args = parser.parse_args()

    start_time = perf_counter()
    solver = random.choice(available_solvers)
    solution = solve_qp2(P, q, G, h, A, b, lb, ub, solver=args.solver)
    end_time = perf_counter()

    print("========================= PRIMAL PROBLEM =========================")
    print("")
    print("    min. 1/2 x^T P x + q^T x")
    print("    s.t.     G x <= h")
    print("             A x == b")
    print("         lb <= x <= ub")
    print("")
    print_matrix_vector(P, "P", q, "q")
    print("")
    print_matrix_vector(G, "G", h, "h")
    print("")
    print_matrix_vector(A, "A", b, "b")
    print("")
    print_matrix_vector(lb.reshape((3, 1)), "lb", ub, "ub")
    print("")

    print("============================ SOLUTION ============================")
    print("")
    print(f"Found in {1e6 * (end_time - start_time):.0f} [us] with {solver}")
    print("")
    print(f"Primal: x = {solution.x}")
    print(f"Dual (G x <= h): z = {solution.z}")
    print(f"Dual (A x == b): y = {solution.y}")
    print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")
    print("")

    dual_feasibility = np.linalg.norm(
        P.dot(solution.x)
        + G.T.dot(solution.z)
        + A.T.dot(solution.y)
        + solution.z_box
        + q,
        np.inf,
    )

    print("===================== OPTIMALITY CONDITIONS =====================")
    print("")
    print("Dual feasibility:")
    print(
        f"    || P x + G^T z + A^T y + z_box + q || = {dual_feasibility:.1e}"
    )
