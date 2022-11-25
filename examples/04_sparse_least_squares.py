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
Test a random sparse QP solver on a sparse least-squares problem.

See also: https://stackoverflow.com/a/74415546/3721564
"""

import random
from time import perf_counter

import numpy as np
import scipy.sparse as spa

import qpsolvers
from qpsolvers import solve_ls

n = 150_000

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
a_dozen_rows = np.linspace(0, n - 1, 12, dtype=int)
G = G[a_dozen_rows]
h = np.ones(12)

# such that sum(x) == 42
A = spa.csc_matrix(np.ones((1, n)))
b = np.array([42.0]).reshape((1,))

# such that x >= 0
lb = np.zeros(n)


if __name__ == "__main__":
    solver = (
        "osqp"
        if "osqp" in qpsolvers.sparse_solvers
        else random.choice(qpsolvers.sparse_solvers)
    )

    start_time = perf_counter()
    x = solve_ls(
        R,
        s,
        G,
        h,
        A,
        b,
        lb,
        solver=solver,
        verbose=False,
    )
    end_time = perf_counter()
    duration_ms = 1e3 * (end_time - start_time)
    tol = 1e-6  # tolerance for checks

    print("")
    print("    min.  || x - s ||^2")
    print("    s.t.  G * x <= h")
    print("          sum(x) = 42")
    print("          0 <= x")
    print("")
    print(f"Found solution in {duration_ms:.0f} milliseconds with {solver}")
    print("")
    print(f"- Objective: {0.5 * (x - s).dot(x - s):.1f}")
    print(f"- G * x <= h: {(G.dot(x) <= h + tol).all()}")
    print(f"- x >= 0: {(x + tol >= 0.0).all()}")
    print(f"- sum(x) = {x.sum():.1f}")
    print("")
