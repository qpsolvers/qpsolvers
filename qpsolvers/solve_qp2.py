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

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .exceptions import NoSolverSelected, SolverNotFound
from .problem import Problem
from .solution import Solution
from .solvers import available_solvers, proxqp_solve_qp2, solve_function


def solve_qp2(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
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
    if sym_proj:
        P = 0.5 * (P + P.transpose())
        warnings.warn(
            "The `sym_proj` feature is deprecated "
            "and will be removed in qpsolvers v2.9",
            DeprecationWarning,
            stacklevel=2,
        )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    problem.check_constraints()
    kwargs["initvals"] = initvals
    kwargs["verbose"] = verbose
    assert solver == "proxqp"
    return proxqp_solve_qp2(P, q, G, h, A, b, lb, ub, **kwargs)
    try:
        return solve_function[solver](
            problem.P,
            problem.q,
            problem.G,
            problem.h,
            problem.A,
            problem.b,
            problem.lb,
            problem.ub,
            **kwargs,
        )
    except KeyError as e:
        raise SolverNotFound(
            f"solver '{solver}' is not in the list "
            f"{available_solvers} of available solvers"
        ) from e
