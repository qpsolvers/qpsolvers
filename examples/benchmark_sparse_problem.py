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
Test all available QP solvers on a sparse quadratic program.
"""

import numpy as np
import scipy.sparse

from IPython import get_ipython
from numpy.linalg import norm
from os.path import basename
from scipy.sparse import csc_matrix

from qpsolvers import dense_solvers, sparse_solvers
from qpsolvers import solve_qp


n = 500
M = scipy.sparse.lil_matrix(scipy.sparse.eye(n))
for i in range(1, n - 1):
    M[i, i + 1] = -1
    M[i, i - 1] = 1
P = csc_matrix(M.dot(M.transpose()))
q = -np.ones((n,))
G = csc_matrix(-scipy.sparse.eye(n))
h = -2 * np.ones((n,))
P_array = np.array(P.todense())
G_array = np.array(G.todense())


def check_same_solutions(tol=0.05):
    sol0 = solve_qp(P, q, G, h, solver=sparse_solvers[0])
    for solver in sparse_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
        relvar = norm(sol - sol0) / norm(sol0)
        assert (
            relvar < tol
        ), f"{solver}'s solution offset by {100.0 * relvar:.1f}%"
    for solver in dense_solvers:
        sol = solve_qp(P_array, q, G_array, h, solver=solver)
        relvar = norm(sol - sol0) / norm(sol0)
        assert (
            relvar < tol
        ), f"{solver}'s solution offset by {100.0 * relvar:.1f}%"


def time_dense_solvers():
    instructions = {
        solver: f"u = solve_qp(P_array, q, G_array, h, solver='{solver}')"
        for solver in dense_solvers
    }
    print("\nDense solvers\n-------------")
    for solver, instr in instructions.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")


def time_sparse_solvers():
    instructions = {
        solver: f"u = solve_qp(P, q, G, h, solver='{solver}')"
        for solver in sparse_solvers
    }
    print("\nSparse solvers\n--------------")
    for solver, instr in instructions.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "Run the benchmark with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()
    print("\nTesting all QP solvers on a sparse quadratic program...")
    check_same_solutions()
    time_dense_solvers()
    time_sparse_solvers()
