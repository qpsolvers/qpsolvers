#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2020 Stephane Caron <stephane.caron@normalesup.org>
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

"""Solver interface for CVXPY"""

from typing import Optional

from cvxpy import Constant, Minimize, Problem, Variable, quad_form
from numpy import array, ndarray


def cvxpy_solve_qp(
    P,
    q,
    G=None,
    h=None,
    A=None,
    b=None,
    initvals=None,
    solver: str = None,
    verbose: bool = False,
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = h
        \\end{array}\\end{split}

    calling a given solver using the `CVXPY <http://www.cvxpy.org/>`_ modelling
    language.

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    initvals : array, shape=(n,), optional
        Warm-start guess vector (not used).
    solver : string, optional
        Solver name in ``cvxpy.installed_solvers()``.
    verbose : bool, optional
        Set to `True` to print out extra information.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        print("CVXPY: note that warm-start values are ignored by wrapper")
    n = q.shape[0]
    x = Variable(n)
    P = Constant(P)  # see http://www.cvxpy.org/en/latest/faq/
    objective = Minimize(0.5 * quad_form(x, P) + q @ x)
    constraints = []
    if G is not None:
        constraints.append(G @ x <= h)
    if A is not None:
        constraints.append(A @ x == b)
    prob = Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)
    if x.value is None:
        return None
    return array(x.value).reshape((n,))
