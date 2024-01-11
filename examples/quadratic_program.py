#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test the "quadprog" QP solver on a small dense problem."""

import random
from time import perf_counter

import numpy as np

from qpsolvers import available_solvers, print_matrix_vector, solve_qp

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = np.dot(M.T, M)  # this is a positive definite matrix
q = np.dot(np.array([3.0, 2.0, 3.0]), M)
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0])
A = np.array([1.0, 1.0, 1.0])
b = np.array([1.0])

x_sol = np.array([0.3076923111580727, -0.6923076888419274, 1.3846153776838548])

if __name__ == "__main__":
    start_time = perf_counter()
    solver = random.choice(available_solvers)
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
    print(f"It should be close to x* = {x_sol}")
    print(f"Found in {1e6 * (end_time - start_time):.0f} [us] with {solver}")
