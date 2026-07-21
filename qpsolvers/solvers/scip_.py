#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Solver interface for `SCIP <https://github.com/scipopt/PySCIPOpt/>`__.

SCIP is currently one of the fastest academically developed solvers for
mixed integer programming (MIP) and mixed integer nonlinear programming
(MINLP). In addition, SCIP provides a highly flexible framework for
constraint integer programming and branch-cut-and-price. It allows for
total control of the solution process and the access of detailed
information down to the guts of the solver. If you are using SCIP in a
scientific work, consider citing the corresponding paper
[Achterberg2009]_.

SCIP solves quadratic programs by branch-and-bound and does not return
dual multipliers: the solution's dual attributes are left to ``None``.

**Warm-start:** this solver interface supports warm starting 🔥
"""

import time
import warnings
from typing import Optional, Union

import numpy as np
import pyscipopt
import scipy.sparse as spa

from ..exceptions import ParamError
from ..problem import Problem
from ..solution import Solution
from ..solve_unconstrained import solve_unconstrained

# Tighten tolerances to QP-grade accuracy (overridable by keyword
# arguments); values below 1e-9 require SCIP to be built with GMP.
DEFAULT_PARAMS = {
    "numerics/feastol": 1e-9,
    "numerics/dualfeastol": 1e-9,
    "limits/gap": 1e-9,
    "limits/absgap": 1e-9,
}


def scip_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using PySCIPOpt.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Raises
    ------
    ParamError
        If a keyword argument is not a valid SCIP parameter.

    Notes
    -----
    Keyword arguments are forwarded to SCIP as parameters. Parameter
    names contain slashes, so they are passed by dictionary unpacking:
    for instance, ``scip_solve_problem(problem, **{"limits/time": 10})``
    sets a time limit of 10 seconds. SCIP parameters include the
    following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``limits/gap``
         - Relative gap between primal and dual bounds at which the
           solve stops.
       * - ``limits/time``
         - Run time limit in seconds.
       * - ``numerics/dualfeastol``
         - Dual feasibility tolerance.
       * - ``numerics/feastol``
         - Primal feasibility tolerance.

    Check out the `SCIP parameter reference
    <https://www.scipopt.org/doc/html/PARAMETERS.php>`_ for all
    available SCIP parameters.

    By default, this interface tightens SCIP's accuracy parameters to
    ``numerics/feastol=1e-9``, ``numerics/dualfeastol=1e-9``,
    ``limits/gap=1e-9`` and ``limits/absgap=1e-9``. Pass any of these
    keys as keyword arguments to override them; for instance,
    ``**{"numerics/feastol": 1e-6, "numerics/dualfeastol": 1e-7,
    "limits/gap": 0.0, "limits/absgap": 0.0}`` restores stock SCIP
    behavior. Lower values for primal or dual tolerances yield more
    precise solutions at the cost of computation time. See *e.g.*
    [Caron2022]_ for a primer of solver tolerances.
    """
    if problem.is_unconstrained:
        warnings.warn(
            "QP is unconstrained: solving with SciPy's LSQR rather than SCIP"
        )
        return solve_unconstrained(problem)

    start_build = time.perf_counter()
    model = pyscipopt.Model()
    model.hideOutput(not verbose)
    for option, value in {**DEFAULT_PARAMS, **kwargs}.items():
        try:
            model.setParam(option, value)
        except KeyError as exc:
            raise ParamError(f'Unknown SCIP parameter "{option}"') from exc

    P, q, G, h, A, b, lb, ub = problem.unpack_as_dense()
    num_vars = P.shape[0]
    x = model.addMatrixVar(
        (num_vars,),
        lb=-model.infinity() if lb is None else lb,
        ub=model.infinity() if ub is None else ub,
    )
    if G is not None:
        model.addMatrixCons(G @ x <= h)
    if A is not None:
        model.addMatrixCons(A @ x == b)

    # SCIP requires a linear objective: minimize t s.t. objective <= t
    t = model.addVar(lb=-model.infinity(), obj=1.0)
    model.addCons(0.5 * (x @ P @ x) + q @ x <= t)

    if initvals is not None:
        x_init = np.asarray(initvals, dtype=float).ravel()
        if x_init.shape[0] != num_vars:
            raise ValueError(
                f"warm-start guess has {x_init.shape[0]} values "
                f"but the problem has {num_vars} variables"
            )
        init_sol = model.createSol()
        for i, val in enumerate(x_init):
            model.setSolVal(init_sol, x[i], val)
        # The warm start must include t, padded so that rounding cannot
        # make the epigraph constraint infeasible
        obj_init = 0.5 * x_init @ P @ x_init + q @ x_init
        model.setSolVal(init_sol, t, obj_init + 1e-8 * (1.0 + abs(obj_init)))
        model.addSol(init_sol, free=True)
    build_time = time.perf_counter() - start_build

    start_solve = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_solve

    solution = Solution(problem)
    solution.build_time = build_time
    solution.solve_time = solve_time
    status = model.getStatus()
    solution.extras = {
        "status": status,
        "primal_bound": model.getPrimalbound(),
        "dual_bound": model.getDualbound(),
    }
    # gaplimit is a success path here since DEFAULT_PARAMS sets gap limits
    solution.found = status in ("optimal", "gaplimit")
    if solution.found:
        x_opt = np.asarray(model.getVal(x), dtype=float)
        solution.x = x_opt
        solution.obj = float(0.5 * x_opt @ P @ x_opt + q @ x_opt)
    return solution


def scip_solve_qp(
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
    r"""Solve a quadratic program using SCIP.

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

    It is solved using `SCIP <https://scipopt.org/>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
    q :
        Cost vector.
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
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ParamError
        If a keyword argument is not a valid SCIP parameter.

    Notes
    -----
    Keyword arguments are forwarded to SCIP as parameters. Parameter
    names contain slashes, so they are passed by dictionary unpacking:
    for instance, ``scip_solve_qp(P, q, G, h, **{"limits/time": 10})``
    sets a time limit of 10 seconds. SCIP parameters include the
    following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``limits/gap``
         - Relative gap between primal and dual bounds at which the
           solve stops.
       * - ``limits/time``
         - Run time limit in seconds.
       * - ``numerics/dualfeastol``
         - Dual feasibility tolerance.
       * - ``numerics/feastol``
         - Primal feasibility tolerance.

    Check out the `SCIP parameter reference
    <https://www.scipopt.org/doc/html/PARAMETERS.php>`_ for all
    available SCIP parameters.

    By default, this interface tightens SCIP's accuracy parameters to
    ``numerics/feastol=1e-9``, ``numerics/dualfeastol=1e-9``,
    ``limits/gap=1e-9`` and ``limits/absgap=1e-9``. Pass any of these
    keys as keyword arguments to override them; for instance,
    ``**{"numerics/feastol": 1e-6, "numerics/dualfeastol": 1e-7,
    "limits/gap": 0.0, "limits/absgap": 0.0}`` restores stock SCIP
    behavior. Lower values for primal or dual tolerances yield more
    precise solutions at the cost of computation time. See *e.g.*
    [Caron2022]_ for a primer of solver tolerances.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = scip_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
