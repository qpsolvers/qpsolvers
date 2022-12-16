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
Solver interface for `qpOASES <https://github.com/coin-or/qpOASES>`__.

qpOASES is an open-source C++ implementation of the online active set strategy,
which was inspired by observations from the field of parametric quadratic
programming. It has theoretical features that make it suitable to model
predictive control. Further numerical modifications have made qpOASES a
reliable QP solver, even when tackling semi-definite, ill-posed or degenerated
QP problems. If you are using qpOASES in some academic work, consider citing
the corresponding paper [Ferreau2014]_.

See the :ref:`installation page <qpoases-install>` for additional instructions
on installing this solver.
"""

import warnings
from typing import Any, List, Optional

import numpy as np
from numpy import array, hstack, ones, vstack
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyQProblem as QProblem
from qpoases import PyQProblemB as QProblemB
from qpoases import PyReturnValue as ReturnValue

from ..problem import Problem
from ..solution import Solution

__infty__ = 1e10


# Return codes not wrapped in qpoases.PyReturnValue
RET_INIT_FAILED = 33
RET_INIT_FAILED_TQ = 34
RET_INIT_FAILED_CHOLESKY = 35
RET_INIT_FAILED_HOTSTART = 36
RET_INIT_FAILED_INFEASIBILITY = 37
RET_INIT_FAILED_UNBOUNDEDNESS = 38
RET_INIT_FAILED_REGULARISATION = 39


def __prepare_options(
    verbose: bool,
    predefined_options: Optional[str],
    **kwargs,
) -> Options:
    """
    Prepare options for qpOASES.

    Parameters
    ----------
    verbose :
        Set to `True` to print out extra information.
    predefined_options :
        Set solver options to one of the pre-defined choices provided by
        qpOASES: ``["default", "fast", "mpc", "reliable"]``.

    Returns
    -------
    :
        Options for qpOASES.
    """
    options = Options()

    # Start from pre-defined options
    if predefined_options is None:
        pass
    elif predefined_options == "fast":
        options.setToFast()
    elif predefined_options == "default":
        options.setToDefault()
    elif predefined_options == "mpc":
        options.setToMPC()
    elif predefined_options == "reliable":
        options.setToReliable()
    else:
        raise ValueError(
            f"unknown qpOASES pre-defined options {predefined_options}'"
        )

    # Override options with explicit ones
    options.printLevel = PrintLevel.MEDIUM if verbose else PrintLevel.NONE
    for param, value in kwargs.items():
        setattr(options, param, value)

    return options


def qpoases_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    max_wsr: int = 1000,
    time_limit: Optional[float] = None,
    predefined_options: Optional[str] = None,
    **kwargs,
) -> Solution:
    """
    Solve a quadratic program using `qpOASES
    <https://github.com/coin-or/qpOASES>`__.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.
    max_wsr :
        Maximum number of Working-Set Recalculations given to qpOASES.
    time_limit :
        Set a run time limit in seconds.
    predefined_options :
        Set solver options to one of the pre-defined choices provided by
        qpOASES, to pick in ``["default", "fast", "mpc", "reliable"]``.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ValueError :
        If ``predefined_options`` is not a valid choice.

    Notes
    -----
    This function relies on an update to qpOASES to allow empty bounds (`lb`,
    `ub`, `lbA` or `ubA`) in Python. This is possible in the C++ API but not by
    the Python API (as of version 3.2.0). If using them, be sure to update the
    Cython file (`qpoases.pyx`) in your distribution of qpOASES to convert
    ``None`` to the null pointer. Check out the `installation instructions
    <https://scaron.info/doc/qpsolvers/installation.html#qpoases>`_.

    Keyword arguments are forwarded as options to qpOASES. For instance, we can
    call ``qpoases_solve_qp(P, q, G, h, u, terminationTolerance=1e-14)``.
    qpOASES options include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``boundRelaxation``
         - Initial relaxation of bounds to start homotopy and initial value for
           far bounds.
       * - ``epsNum``
         - Numerator tolerance for ratio tests.
       * - ``epsDen``
         - Denominator tolerance for ratio tests.
       * - ``numRefinementSteps``
         - Maximum number of iterative refinement steps.
       * - ``terminationTolerance``
         - Relative termination tolerance to stop homotopy.

    Check out pages 28 to 30 of `qpOASES User's Manual
    <https://www.coin-or.org/qpOASES/doc/3.1/manual.pdf>`_. for all available
    options.
    """
    if initvals is not None:
        print("qpOASES: note that warm-start values ignored by wrapper")
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n = P.shape[0]
    lb_C: Optional[np.ndarray] = None
    C = np.array([])
    if G is not None and h is not None:
        if A is not None and b is not None:
            C = vstack([G, A])
            lb_C = hstack([-__infty__ * ones(h.shape[0]), b])
            ub_C = hstack([h, b])
        else:  # no equality constraint
            C = G
            ub_C = h
    else:  # no inequality constraint
        if A is not None and b is not None:
            C = A
            lb_C = b
            ub_C = b

    args: List[Any] = []
    if C.shape[0] > 0:
        qp = QProblem(n, C.shape[0])
        args = [P, q, C, lb, ub, lb_C, ub_C, array([max_wsr])]
    else:  # at most box constraints
        qp = QProblemB(n)
        args = [P, q, lb, ub, array([max_wsr])]
    if time_limit is not None:
        args.append(array([time_limit]))

    options = __prepare_options(verbose, predefined_options, **kwargs)
    qp.setOptions(options)
    return_value = qp.init(*args)

    solution = Solution(problem)
    if RET_INIT_FAILED <= return_value <= RET_INIT_FAILED_REGULARISATION:
        return solution
    if return_value == ReturnValue.MAX_NWSR_REACHED:
        warnings.warn(f"qpOASES reached the maximum number of WSR ({max_wsr})")
        return solution
    x_opt = np.empty((n,))
    z_opt = np.empty((n + C.shape[0],))
    qp.getPrimalSolution(x_opt)  # can't return RET_QP_NOT_SOLVED at this point
    qp.getDualSolution(z_opt)
    solution.x = x_opt
    m = G.shape[0] if G is not None else 0
    if lb is not None or ub is not None:
        solution.z_box = -z_opt[:n]
    if G is not None:
        solution.z = -z_opt[n : n + m]
    if A is not None:
        solution.y = -z_opt[n + m : n + m + A.shape[0]]
    solution.obj = qp.getObjVal()
    return solution


def qpoases_solve_qp(
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
    max_wsr: int = 1000,
    time_limit: Optional[float] = None,
    predefined_options: Optional[str] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `qpOASES <https://github.com/coin-or/qpOASES>`__.

    Parameters
    ----------
    P :
        Symmetric cost matrix.
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
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.
    max_wsr :
        Maximum number of Working-Set Recalculations given to qpOASES.
    time_limit :
        Set a run time limit in seconds.
    predefined_options :
        Set solver options to one of the pre-defined choices provided by
        qpOASES, to pick in ``["default", "fast", "mpc", "reliable"]``.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ValueError :
        If ``predefined_options`` is not a valid choice.

    Notes
    -----
    This function relies on an update to qpOASES to allow empty bounds (`lb`,
    `ub`, `lbA` or `ubA`) in Python. This is possible in the C++ API but not by
    the Python API (as of version 3.2.0). If using them, be sure to update the
    Cython file (`qpoases.pyx`) in your distribution of qpOASES to convert
    ``None`` to the null pointer. Check out the `installation instructions
    <https://scaron.info/doc/qpsolvers/installation.html#qpoases>`_.

    Keyword arguments are forwarded as options to qpOASES. For instance, we can
    call ``qpoases_solve_qp(P, q, G, h, u, terminationTolerance=1e-14)``.
    qpOASES options include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``boundRelaxation``
         - Initial relaxation of bounds to start homotopy and initial value for
           far bounds.
       * - ``epsNum``
         - Numerator tolerance for ratio tests.
       * - ``epsDen``
         - Denominator tolerance for ratio tests.
       * - ``numRefinementSteps``
         - Maximum number of iterative refinement steps.
       * - ``terminationTolerance``
         - Relative termination tolerance to stop homotopy.

    Check out pages 28 to 30 of `qpOASES User's Manual
    <https://www.coin-or.org/qpOASES/doc/3.1/manual.pdf>`_. for all available
    options.
    """
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpoases_solve_problem(
        problem,
        initvals,
        verbose,
        max_wsr,
        time_limit,
        predefined_options,
        **kwargs,
    )
    return solution.x
