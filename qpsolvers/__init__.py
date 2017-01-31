#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# qpsolvers. If not, see <http://www.gnu.org/licenses/>.

"""
Solvers with matrix-vector input
"""

try:  # CVXOPT
    from backend_cvxopt import cvxopt_solve_qp
    from backend_cvxopt import mosek_solve_qp
except ImportError:
    def cvxopt_solve_qp(*args, **kwargs):
        raise ImportError("CVXOPT not found")

try:  # quadprog
    from backend_quadprog import quadprog_solve_qp
except ImportError:
    def quadprog_solve_qp(*args, **kwargs):
        raise ImportError("quadprog not found")

try:  # qpOASES
    from backend_qpoases import qpoases_solve_qp
except ImportError:
    def qpoases_solve_qp(*args, **kwargs):
        raise ImportError("qpOASES not found")

"""
Solvers with symbolic input (NB: problem creation takes time)
"""

try:  # CVXPY
    from backend_cvxpy import cvxpy_solve_qp
except ImportError:
    def cvxpy_solve_qp(*args, **kwargs):
        raise ImportError("CVXPY not found")

try:  # Gurobi
    from backend_gurobi import gurobi_solve_qp
except ImportError:
    def gurobi_solve_qp(*args, **kwargs):
        raise ImportError("Gurobi not found")

__all__ = [
    'cvxopt_solve_qp',
    'cvxpy_solve_qp',
    'gurobi_solve_qp',
    'mosek_solve_qp',
    'qpoases_solve_qp',
    'quadprog_solve_qp',
]


def solve_qp(P, q, G=None, h=None, A=None, b=None, solver='quadprog',
             initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using one of the available QP solvers.

    Parameters
    ----------
    P : array
        Quadratic cost matrix.
    q : array
        Quadratic cost vector.
    G : array
        Linear inequality matrix.
    h : array
        Linear inequality vector.
    A : array
        Linear equality matrix.
    b : array
        Linear equality vector.
    solver : str
        Name of the solver to use, to choose in ['cvxopt', 'cvxpy', 'gurobi',
        'mosek', 'qpoases', 'quadprog'].
    initvals : array
        Vector of initial `x` values used to warm-start the solver.

    Returns
    -------
    x : array or None
        Optimal solution if found, None otherwise.
    """
    if solver == 'cvxopt':
        return cvxopt_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'cvxpy':
        return cvxpy_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'gurobi':
        return gurobi_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'mosek':
        return mosek_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'qpoases':
        return qpoases_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'quadprog':
        return quadprog_solve_qp(P, q, G, h, A, b, initvals=initvals)
    raise Exception("solver '%s' not recognized" % solver)
