#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test a random available QP solver on a small least-squares problem."""

import random
from time import perf_counter

import numpy as np

import qpsolvers
from qpsolvers import print_matrix_vector, solve_ls

R = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
s = np.array([3.0, 2.0, 3.0])
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0])

x_sol = np.array([0.1299734765610818, -0.0649867382805409, 1.7400530468778364])

if __name__ == "__main__":
    start_time = perf_counter()
    solver = random.choice(qpsolvers.available_solvers)
    x = solve_ls(R, s, G, h, solver=solver, verbose=False)
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
    print(f"It should be close to x* = {x_sol}")
    print(f"Found in {1e6 * (end_time - start_time):.0f} [us] with {solver}")
