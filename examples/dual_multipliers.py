#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Get both primal and dual solutions to a quadratic program."""

import random

import numpy as np

from qpsolvers import (
    Problem,
    available_solvers,
    print_matrix_vector,
    solve_problem,
)

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = np.dot(M.T, M)  # this is a positive definite matrix
q = np.dot(np.array([3.0, 2.0, 3.0]), M)
G = np.array([[4.0, 2.0, 0.0], [-1.0, 2.0, -1.0]])
h = np.array([1.0, -2.0])
A = np.array([1.0, 1.0, 1.0]).reshape((1, 3))
b = np.array([1.0])
lb = np.array([-0.5, -0.4, -0.5])
ub = np.array([1.0, 1.0, 1.0])


if __name__ == "__main__":
    solver = random.choice(available_solvers)
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = solve_problem(problem, solver)

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
    print(f'Found with solver="{solver}"')
    print("")
    print_matrix_vector(
        solution.x.reshape((3, 1)),
        "Primal x*",
        solution.z,
        "Dual (Gx <= h) z*",
    )
    print("")
    print_matrix_vector(
        solution.y.reshape((1, 1)),
        "Dual (Ax == b) y*",
        solution.z_box.reshape((3, 1)),
        "Dual (lb <= x <= ub) z_box*",
    )
    print("")

    print("=== Optimality  checks ===")
    print(f"- Primal residual: {solution.primal_residual():.1e}")
    print(f"- Dual residual:   {solution.dual_residual():.1e}")
    print(f"- Duality gap:     {solution.duality_gap():.1e}")
    print("")
