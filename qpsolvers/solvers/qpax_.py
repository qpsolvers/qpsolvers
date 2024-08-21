#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 Lev Kozlov
"""Solver interface for `qpax <https://github.com/kevin-tracy/qpax>`__.
qpax is an open source QP solver that can be combined with JAX's jit and vmap
functionality, as well as differentiated with reverse-mode differentiation. It
is based on a primal-dual interior point algorithm. If you are using qpax in
some academic work, consider citing the corresponding paper [Tracy2024]_.
"""

from typing import Optional

import numpy as np

import qpax


from ..problem import Problem
from ..solution import Solution
from ..conversions import linear_from_box_inequalities, split_dual_linear_box


def qpax_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Solution:
    # construct the qpax problem
    G, h = linear_from_box_inequalities(
        problem.G, problem.h, problem.lb, problem.ub, use_sparse=False
    )

    n = problem.q.shape[0]

    # qpax does not support A, b to be None.
    A_qpax = np.zeros((0, n)) if problem.A is None else problem.A
    b_qpax = np.zeros((0,)) if problem.b is None else problem.b

    x, s, z, y, converged, iters1 = qpax.solve_qp(
        problem.P,
        problem.q,
        A_qpax,
        b_qpax,
        G,
        h,
        solver_tol=1e-5,
    )

    solution = Solution(problem)
    solution.x = x
    solution.found = converged
    solution.y = y

    # split the dual variables into the box constraints and the linear constraints
    solution.z, solution.z_box = split_dual_linear_box(
        z, problem.lb, problem.ub
    )

    # TODO: fill external solver info
    solution.extras

    return solution


def qpax_solve_qp(
    P, q, G, h, A, b, lb, ub, initvals=None, verbose=False, **kwargs
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using qpax.

    Paper: https://arxiv.org/pdf/2406.11749

    # TODO: copy from paper
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpax_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
