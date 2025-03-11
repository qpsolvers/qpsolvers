#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors
# Copyright 2021 Dustin Kenefake

"""Solver interface for `Gurobi <https://www.gurobi.com/>`__.

The Gurobi Optimizer suite ships several solvers for mathematical programming,
including problems that have linear constraints, bound constraints, integrality
constraints, cone constraints, or quadratic constraints. It targets modern CPU
architectures and multi-core processors,

See the :ref:`installation page <gurobi-install>` for additional instructions
on installing this solver.
"""

import warnings
from typing import Optional, Union

import gurobipy
import numpy as np
import scipy.sparse as spa
from gurobipy import GRB

from ..problem import Problem
from ..solution import Solution


def gurobi_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using Gurobi.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to Gurobi as parameters. For instance, we
    can call ``gurobi_solve_qp(P, q, G, h, u, FeasibilityTol=1e-8,
    OptimalityTol=1e-8)``. Gurobi settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasibilityTol``
         - Primal feasibility tolerance.
       * - ``OptimalityTol``
         - Dual feasibility tolerance.
       * - ``PSDTol``
         - Positive semi-definite tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://www.gurobi.com/documentation/9.5/refman/parameter_descriptions.html>`_
    documentation for all available Gurobi parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    if initvals is not None:
        warnings.warn("warm-start values are ignored by this wrapper")

    model = gurobipy.Model()
    if not verbose:
        model.setParam(GRB.Param.OutputFlag, 0)
    for param, value in kwargs.items():
        model.setParam(param, value)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = spa.eye(num_vars)
    x = model.addMVar(
        num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
    )
    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None:
        ineq_constr = model.addMConstr(G, x, GRB.LESS_EQUAL, h)
    if A is not None:
        eq_constr = model.addMConstr(A, x, GRB.EQUAL, b)
    if lb is not None:
        lb_constr = model.addMConstr(identity, x, GRB.GREATER_EQUAL, lb)
    if ub is not None:
        ub_constr = model.addMConstr(identity, x, GRB.LESS_EQUAL, ub)
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, sense=GRB.MINIMIZE)
    model.optimize()

    solution = Solution(problem)
    solution.extras["status"] = model.status
    solution.found = model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    if solution.found:
        solution.x = x.X
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)
    return solution


def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[gurobipy.MConstr],
    eq_constr: Optional[gurobipy.MConstr],
    lb_constr: Optional[gurobipy.MConstr],
    ub_constr: Optional[gurobipy.MConstr],
) -> None:
    solution.z = -ineq_constr.Pi if ineq_constr is not None else np.empty((0,))
    solution.y = -eq_constr.Pi if eq_constr is not None else np.empty((0,))
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = -ub_constr.Pi - lb_constr.Pi
    elif ub_constr is not None:  # lb_constr is None
        solution.z_box = -ub_constr.Pi
    elif lb_constr is not None:  # ub_constr is None
        solution.z_box = -lb_constr.Pi
    else:  # lb_constr is None and ub_constr is None
        solution.z_box = np.empty((0,))


def gurobi_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using Gurobi.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h                \\
                & A x = b                   \\
                & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `Gurobi <http://www.gurobi.com/>`__.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to Gurobi as parameters. For instance, we
    can call ``gurobi_solve_qp(P, q, G, h, u, FeasibilityTol=1e-8,
    OptimalityTol=1e-8)``. Gurobi settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasibilityTol``
         - Primal feasibility tolerance.
       * - ``OptimalityTol``
         - Dual feasibility tolerance.
       * - ``PSDTol``
         - Positive semi-definite tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://www.gurobi.com/documentation/9.5/refman/parameter_descriptions.html>`_
    documentation for all available Gurobi parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = gurobi_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
