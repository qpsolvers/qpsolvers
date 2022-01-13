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

"""Quadratic programming solvers in Python with a unified API"""

from .solve_ls import solve_ls
from .solve_qp import solve_qp
from .solve_qp import solve_safer_qp
from .solvers import available_solvers
from .solvers import cvxopt_solve_qp
from .solvers import cvxpy_solve_qp
from .solvers import dense_solvers
from .solvers import ecos_solve_qp
from .solvers import gurobi_solve_qp
from .solvers import mosek_solve_qp
from .solvers import osqp_solve_qp
from .solvers import qpoases_solve_qp
from .solvers import quadprog_solve_qp
from .solvers import scs_solve_qp
from .solvers import sparse_solvers
from .utils import print_matrix_vector

__version__ = "1.8.0"

__all__ = [
    "__version__",
    "available_solvers",
    "cvxopt_solve_qp",
    "cvxpy_solve_qp",
    "dense_solvers",
    "ecos_solve_qp",
    "gurobi_solve_qp",
    "mosek_solve_qp",
    "osqp_solve_qp",
    "print_matrix_vector",
    "qpoases_solve_qp",
    "quadprog_solve_qp",
    "scs_solve_qp",
    "solve_ls",
    "solve_qp",
    "solve_safer_qp",
    "sparse_solvers",
]
