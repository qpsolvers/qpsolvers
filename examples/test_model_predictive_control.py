#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test all available QP solvers on a model predictive control problem."""

from os.path import basename

from IPython import get_ipython
from qpsolvers import dense_solvers, sparse_solvers

from model_predictive_control import (
    HumanoidModelPredictiveControl,
    HumanoidSteppingProblem,
)

problem = HumanoidSteppingProblem()
mpc = HumanoidModelPredictiveControl(problem)


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "This example should be run with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()

    dense_instr = {
        solver: f"u = mpc.solve(solver='{solver}', sparse=False)"
        for solver in dense_solvers
    }
    sparse_instr = {
        solver: f"u = mpc.solve(solver='{solver}', sparse=True)"
        for solver in sparse_solvers
    }

    benchmark = "https://github.com/qpsolvers/qpbenchmark"
    print("\nTesting QP solvers on one given model predictive control problem")
    print(f"For a proper benchmark, check out {benchmark}")

    print("\nDense solvers\n-------------")
    for solver, instr in dense_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().run_line_magic("timeit", instr)

    print("\nSparse solvers\n--------------")
    for solver, instr in sparse_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().run_line_magic("timeit", instr)
