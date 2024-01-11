#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test all available QP solvers on a sparse quadratic program."""

from os.path import basename

import numpy as np
import scipy.sparse
from IPython import get_ipython
from numpy.linalg import norm
from scipy.sparse import csc_matrix

from qpsolvers import dense_solvers, solve_qp, sparse_solvers

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
        get_ipython().run_line_magic("timeit", instr)


def time_sparse_solvers():
    instructions = {
        solver: f"u = solve_qp(P, q, G, h, solver='{solver}')"
        for solver in sparse_solvers
    }
    print("\nSparse solvers\n--------------")
    for solver, instr in instructions.items():
        print(f"{solver}: ", end="")
        get_ipython().run_line_magic("timeit", instr)


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "This example should be run with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()

    benchmark = "https://github.com/qpsolvers/qpbenchmark"
    print("\nTesting all QP solvers on one given sparse quadratic program")
    print(f"For a proper benchmark, check out {benchmark}")

    check_same_solutions()
    time_dense_solvers()
    time_sparse_solvers()
