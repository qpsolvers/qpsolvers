#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of oqp.
#
# oqp is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# oqp is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# oqp. If not, see <http://www.gnu.org/licenses/>.

"""
Solvers with matrix-vector input
"""

try:  # CVXOPT
    from backend_cvxopt import cvxopt_solve_qp
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
    'quadprog_solve_qp',
    'qpoases_solve_qp',
]
