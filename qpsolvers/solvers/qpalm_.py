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

"""Solver interface for `QPALM`_.

.. _QPALM: https://github.com/kul-optec/QPALM

C implementation of QPALM, a proximal augmented Lagrangian based solver for
(possibly nonconvex) quadratic programs. If you use QPALM in some
academic work, consider citing the corresponding paper [Hermans2022]_.
"""

from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from ..exceptions import ParamError, ProblemError
from ..problem import Problem
from ..solution import Solution


def qpalm_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using QPALM.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Raises
    ------
    ParamError
        If a warm-start value is given both in `initvals` and the `x` keyword
        argument.
    """
    if initvals is not None:
        if "x" in kwargs:
            raise ParamError(
                "Warm-start value specified in both `initvals` and `x` kwargs"
            )
        kwargs["x"] = initvals
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]
    use_csc: bool = (
        not isinstance(P, np.ndarray)
        or (G is not None and not isinstance(G, np.ndarray))
        or (A is not None and not isinstance(A, np.ndarray))
    )
    C_prox, u_prox, l_prox = __combine_inequalities(G, h, lb, ub, n, use_csc)
    result = solve(
        P,
        q,
        A,
        b,
        C_prox,
        l_prox,
        u_prox,
        verbose=verbose,
        **kwargs,
    )
    solution = Solution(problem)
    solution.extras = {"info": result.info}
    solution.x = result.x
    solution.y = result.y
    if lb is not None or ub is not None:
        solution.z = result.z[:-n]
        solution.z_box = result.z[-n:]
    else:  # lb is None and ub is None
        solution.z = result.z
    return solution


def qpalm_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    lb: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    ub: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using QPALM.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `QPALM <https://github.com/kul-optec/QPALM>`__.

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
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpalm_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
