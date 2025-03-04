#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023 Inria

"""Solve an unconstrained quadratic program."""

import numpy as np
from scipy.sparse.linalg import lsqr

from .exceptions import ProblemError
from .problem import Problem
from .solution import Solution


def solve_unconstrained(problem: Problem) -> Solution:
    """Solve an unconstrained quadratic program with SciPy's LSQR.

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
