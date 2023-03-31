#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""Solve an unconstrained quadratic program."""

import numpy as np
from scipy.sparse.linalg import lsqr

from .exceptions import ProblemError
from .problem import Problem
from .solution import Solution


def solve_unconstrained(problem: Problem) -> Solution:
    """Solve an unconstrained quadratic program.

    Parameters
    ----------
    problem :
        Unconstrained quadratic program.

    Returns
    -------
    :
        Solution to the unconstrained QP, if it is bounded.

    Raises
    ------
    ValueError
        If the quadratic program is not unbounded below.
    """
    P, q, _, _, _, _, _, _ = problem.unpack()
    solution = Solution(problem)
    solution.x = lsqr(P, -q)[0]
    cost_check = np.linalg.norm(P @ solution.x + q)
    if cost_check > 1e-8:
        raise ProblemError(
            f"problem is unbounded below (cost_check={cost_check:.1e}), "
            "q has component in the nullspace of P"
        )
    solution.found = True
    return solution
