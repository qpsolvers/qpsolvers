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
Solve quadratic programs.
"""

from typing import Optional

import numpy as np

from .exceptions import SolverNotFound
from .problem import Problem
from .solution import Solution
from .solvers import available_solvers, solve_function


def solve_problem(
    problem: Problem,
    solver: str,
    initvals: Optional[np.ndarray] = None,
    sym_proj: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    if solver is None:
        raise NoSolverSelected(
            "Set the `solver` keyword argument to one of the "
            f"available solvers in {available_solvers}"
        )
    problem.check_constraints()
    kwargs["initvals"] = initvals
    kwargs["verbose"] = verbose
    try:
        return solve_function[solver](problem, **kwargs)
    except KeyError as e:
        raise SolverNotFound(
            f"solver '{solver}' is not in the list "
            f"{available_solvers} of available solvers"
        ) from e
