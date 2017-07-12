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

available_solvers = []
matrix_solvers = []
symbolic_solvers = []


# CVXOPT
# ======

try:
    from cvxopt_ import cvxopt_solve_qp
    available_solvers.append('cvxopt')
    matrix_solvers.append('cvxopt')
except ImportError:
    def cvxopt_solve_qp(*args, **kwargs):
        raise ImportError("CVXOPT not found")

# CVXPY
# =====

try:
    from cvxpy_ import cvxpy_solve_qp
    available_solvers.append('cvxpy')
    symbolic_solvers.append('cvxpy')
except ImportError:
    def cvxpy_solve_qp(*args, **kwargs):
        raise ImportError("CVXPY not found")

# Gurobi
# ======

try:
    from gurobi_ import gurobi_solve_qp
    available_solvers.append('gurobi')
    symbolic_solvers.append('gurobi')
except ImportError:
    def gurobi_solve_qp(*args, **kwargs):
        raise ImportError("Gurobi not found")

# Mosek
# =====

try:
    from mosek_ import mosek_solve_qp
    available_solvers.append('mosek')
    symbolic_solvers.append('mosek')
except ImportError:
    def mosek_solve_qp(*args, **kwargs):
        raise ImportError("mosek not found")

# OSQP
# ====

try:
    from osqp_ import osqp_solve_qp
    available_solvers.append('osqp')
    matrix_solvers.append('osqp')
except ImportError:
    def osqp_solve_qp(*args, **kwargs):
        raise ImportError("osqp not found")

# qpOASES
# =======

try:
    from qpoases_ import qpoases_solve_qp
    available_solvers.append('qpoases')
    matrix_solvers.append('qpoases')
except ImportError:
    def qpoases_solve_qp(*args, **kwargs):
        raise ImportError("qpOASES not found")

# quadprog
# ========

try:
    from quadprog_ import quadprog_solve_qp
    available_solvers.append('quadprog')
    matrix_solvers.append('quadprog')
except ImportError:
    def quadprog_solve_qp(*args, **kwargs):
        raise ImportError("quadprog not found")


__all__ = [
    'available_solvers',
    'cvxopt_solve_qp',
    'cvxpy_solve_qp',
    'gurobi_solve_qp',
    'matrix_solvers',
    'mosek_solve_qp',
    'qpoases_solve_qp',
    'quadprog_solve_qp',
    'symbolic_solvers',
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
    solver : string, optional
        Name of the solver to use, to choose in ['cvxopt', 'cvxpy', 'gurobi',
        'mosek', 'qpoases', 'quadprog'].
    initvals : array, optional
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
    elif solver == 'osqp':
        return osqp_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'qpoases':
        return qpoases_solve_qp(P, q, G, h, A, b, initvals=initvals)
    elif solver == 'quadprog':
        return quadprog_solve_qp(P, q, G, h, A, b, initvals=initvals)
    raise Exception("solver '%s' not recognized" % solver)
