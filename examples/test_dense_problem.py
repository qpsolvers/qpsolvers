#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test all available QP solvers on a dense quadratic program."""

from os.path import basename

from IPython import get_ipython
from numpy import array, dot
from qpsolvers import dense_solvers, solve_qp, sparse_solvers
from scipy.sparse import csc_matrix

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
            "This example should be run with IPython:\n\n"
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

    benchmark = "https://github.com/qpsolvers/qpbenchmark"
    print("\nTesting all QP solvers on one given dense quadratic program")
    print(f"For a proper benchmark, check out {benchmark}")

    sol0 = solve_qp(P, q, G, h, solver=dense_solvers[0])
    abstol = 2e-4  # tolerance on absolute solution error
    for solver in dense_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
    for solver in sparse_solvers:
        sol = solve_qp(P_csc, q, G_csc, h, solver=solver)

    print("\nDense solvers\n-------------")
    for solver, instr in dense_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().run_line_magic("timeit", instr)

    print("\nSparse solvers\n--------------")
    for solver, instr in sparse_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().run_line_magic("timeit", instr)
