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

import warnings
from typing import Optional

import numpy as np

import qpax


from ..problem import Problem
from ..solution import Solution
from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ParamError, ProblemError


def qpax_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Solution:
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]

    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by PIQP")

    if G is None and h is not None:
        raise ProblemError(
            "Inconsistent inequalities: G is not set but h is set"
        )
    if G is not None and h is None:
        raise ProblemError("Inconsistent inequalities: G is set but h is None")
    if A is None and b is not None:
        raise ProblemError(
            "Inconsistent inequalities: A is not set but b is set"
        )
    if A is not None and b is None:
        raise ProblemError("Inconsistent inequalities: A is set but b is None")

    # construct the qpax problem
    G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
    if G is None:
        G = np.zeros((0, n))
        h = np.zeros((0,))

    # qpax does not support A, b to be None.
    A_qpax = np.zeros((0, n)) if A is None else A
    b_qpax = np.zeros((0)) if b is None else b

    # qpax does not support sparse matrices, so we need to convert them to dense
    P = P.toarray() if hasattr(P, "toarray") else P
    A_qpax = A_qpax.toarray() if hasattr(A_qpax, "toarray") else A_qpax
    G = G.toarray() if hasattr(G, "toarray") else G

    try:
        x, s, z, y, converged, iters1 = qpax.solve_qp(
            P,
            q,
            A_qpax,
            b_qpax,
            G,
            h,
            solver_tol=1e-5,
        )
    except Exception as e:
        raise ProblemError(f"qpax failed with error: {e}")

    if not converged:
        print("Something had happened")

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
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using qpax.

    Paper: https://arxiv.org/pdf/2406.11749

    # TODO: copy from paper
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpax_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
