#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solve quadratic programs."""

from typing import Optional

import numpy as np

from ._internals import available_solvers, solve_function
from .exceptions import SolverNotFound
from .problem import Problem
from .solution import Solution


def solve_problem(
    problem: Problem,
    solver: str,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    r"""Solve a quadratic program using a given solver.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    solver :
        Name of the solver, to choose in :data:`qpsolvers.available_solvers`.
    initvals :
        Primal candidate vector :math:`x` values used to warm-start the solver.
    verbose :
        Set to ``True`` to print out extra information.

    Note
    ----
    In quadratic programming, the matrix :math:`P` should be symmetric. Many
    solvers (including CVXOPT, OSQP and quadprog) assume this is the case and
    may return unintended results when the provided matrix is not. Thus, make
    sure you matrix is indeed symmetric before calling this function, for
    instance by projecting it on its symmetric part :math:`S = \frac{1}{2} (P
    + P^T)`.

    Returns
    -------
    :
        Solution found by the solver, if any, along with solver-specific return
        values.

    Raises
    ------
    SolverNotFound
        If the requested solver is not in :data:`qpsolvers.available_solvers`.

    ValueError
        If the problem is not correctly defined. For instance, if the solver
        requires a definite cost matrix but the provided matrix :math:`P` is
        not.

    Notes
    -----
    Extra keyword arguments given to this function are forwarded to the
    underlying solver. For example, we can call OSQP with a custom absolute
    feasibility tolerance by ``solve_problem(problem, solver='osqp',
    eps_abs=1e-6)``. See the :ref:`Supported solvers <Supported solvers>` page
    for details on the parameters available to each solver.

    There is no guarantee that a ``ValueError`` is raised if the provided
    problem is non-convex, as some solvers don't check for this. Rather, if the
    problem is non-convex and the solver fails because of that, then a
    ``ValueError`` will be raised.
    """
    problem.check_constraints()
    kwargs["initvals"] = initvals
    kwargs["verbose"] = verbose
    try:
        return solve_function[solver](problem, **kwargs)
    except KeyError as e:
        raise SolverNotFound(
            f"found solvers {available_solvers} "
            f"but '{solver}' is not one of them; if '{solver}' is "
            f"listed in https://github.com/qpsolvers/qpsolvers#solvers "
            f"you can run ``pip install qpsolvers[{solver}]``"
        ) from e
