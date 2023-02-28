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

"""Solver interface for `NPPro`__.

The NPPro solver implements a Newton Projection with Proportioning method for convex
quadratic programming. Currently, it is designed for dense problems only, and
convexity is the only assumption it makes on problem data.
"""

import warnings
from typing import Optional, Union

import numpy as np
import nppro

from ..problem import Problem
from ..solution import Solution


def nppro_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using NPPro.

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
        Solution returned by the solver.

    Raises
    ------

    Notes
    -----

    """
    P, q, G, h, A, b, lb, ub = problem.unpack()

    n = P.shape[0]
    m_iq = G.shape[0] if G is not None else 0
    m_eq = A.shape[0] if A is not None else 0
    m = m_iq + m_eq

    A_ = None
    l_ = None
    u_ = None
    lb_ = np.full(q.shape, -np.infty)
    ub_ = np.full(q.shape, +np.infty)
    if G is not None and h is not None:
        A_ = G
        l_ = np.full(h.shape, -np.infty)
        u_ = h
    if A is not None and b is not None:
        A_ = A if A_ is None else np.vstack([A_, A])
        l_ = b if l_ is None else np.hstack([l_, b])
        u_ = b if u_ is None else np.hstack([u_, b])
    if lb is not None:
        lb_ = lb
    if ub is not None:
        ub_ = ub

    # Create solver object
    solver = nppro.CreateSolver(n, m)

    # Set options
    solver.setOption_MaxIter(100);
    solver.setOption_SkipPreprocessing(False);
    solver.setOption_SkipPhaseOne(False);
    solver.setOption_InfVal(1e16);
    solver.setOption_HessianUpdates(True);

    x0 = np.full(q.shape, 0)
    if initvals is not None:
        x0 = initvals

    # Conversion to datatype supported by the solver's C++ interface
    P = np.asarray(P, order='C', dtype=np.float64)
    q = np.asarray(q, order='C', dtype=np.float64)
    A_ = np.asarray(A_, order='C', dtype=np.float64)
    l_ = np.asarray(l_, order='C', dtype=np.float64)
    u_ = np.asarray(u_, order='C', dtype=np.float64)
    lb_ = np.asarray(lb_, order='C', dtype=np.float64)
    ub_ = np.asarray(ub_, order='C', dtype=np.float64)
    x0 = np.asarray(x0, order='C', dtype=np.float64)

    # Call solver
    x, fval, exitflag, iter_ = solver.solve(P, q, A_, l_, u_, lb_, ub_, x0)

    # Store solution
    solution = Solution(problem)
    if exitflag != 0 or np.isnan(x).any():
        # The second condition typically handle positive semi-definite cases
        # that are not catched by the solver yet
        warnings.warn(f"NPPro exited with status {exitflag}")
        return solution
    solution.x = x
    if G is not None:
        solution.z = np.full(h.shape, 0) # not available yet
    if A is not None:
        solution.y = np.full(b.shape, 0) # not available yet
    if lb is not None or ub is not None:
        solution.z_box = np.full(q.shape, 0) # not available yet
    solution.extras = {
        "cost": fval,
        "iter": iter_,
    }
    return solution


def nppro_solve_qp(
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
    r"""Solve a quadratic program using NPPro.

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

    It is solved using `NPPro`__.

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

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------

    Notes
    -----

    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = nppro_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x
