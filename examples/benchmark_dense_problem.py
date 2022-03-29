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
Test all available QP solvers on a dense quadratic program.
"""

from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from os.path import basename
from scipy.sparse import csc_matrix

from qpsolvers import dense_solvers, sparse_solvers
from qpsolvers import solve_qp


M = array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = dot(M.T, M)
q = dot(array([3.0, 2.0, 3.0]), M)
G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = array([3.0, 2.0, -2.0])
P_csc = csc_matrix(P)
G_csc = csc_matrix(G)


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "Run the benchmark with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()

    dense_instr = {
        solver: f"u = solve_qp(P, q, G, h, solver='{solver}')"
        for solver in dense_solvers
    }
    sparse_instr = {
        solver: f"u = solve_qp(P_csc, q, G_csc, h, solver='{solver}')"
        for solver in sparse_solvers
    }

    print("\nTesting all QP solvers on a dense quadratic program...")

    sol0 = solve_qp(P, q, G, h, solver=dense_solvers[0])
    abstol = 2e-4  # tolerance on absolute solution error
    for solver in dense_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
        delta = norm(sol - sol0)
        assert delta < abstol, f"{solver}'s solution offset by {delta:.1e}"
    for solver in sparse_solvers:
        sol = solve_qp(P_csc, q, G_csc, h, solver=solver)
        delta = norm(sol - sol0)
        assert delta < abstol, f"{solver}'s solution offset by {delta:.1e}"

    print("\nDense solvers\n-------------")
    for solver, instr in dense_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")

    print("\nSparse solvers\n--------------")
    for solver, instr in sparse_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")
