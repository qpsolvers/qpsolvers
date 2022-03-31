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
Test all available QP solvers on a model predictive control problem.
"""

from IPython import get_ipython
from os.path import basename

from qpsolvers import dense_solvers, sparse_solvers

from model_predictive_control import HumanoidSteppingProblem
from model_predictive_control import HumanoidModelPredictiveControl

problem = HumanoidSteppingProblem()
mpc = HumanoidModelPredictiveControl(problem)


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "Run the benchmark with IPython:\n\n"
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

    print("\nTesting all QP solvers on a model predictive control problem...")

    print("\nDense solvers\n-------------")
    for solver, instr in dense_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")

    print("\nSparse solvers\n--------------")
    for solver, instr in sparse_instr.items():
        print(f"{solver}: ", end="")
        get_ipython().magic(f"timeit {instr}")
