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

"""Test a random sparse QP solver on a sparse least-squares problem.

See also: https://stackoverflow.com/a/74415546/3721564
"""

import random
from time import perf_counter

import qpsolvers
from qpsolvers import solve_ls
from qpsolvers.problems import get_sparse_least_squares

if __name__ == "__main__":
    solver = random.choice(qpsolvers.sparse_solvers)

    R, s, G, h, A, b, lb, ub = get_sparse_least_squares(n=150_000)
    start_time = perf_counter()
    x = solve_ls(
        R,
        s,
        G,
        h,
        A,
        b,
        lb,
        ub,
        solver=solver,
        verbose=False,
        sparse_conversion=True,
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
