#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stéphane Caron <stephane.caron@normalesup.org>
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

# This file allows the whole repository to act as a Python module when e.g.
# included as a git submodule inside a project. It incurs some redundant
# bookkeeping but some users find it useful.

from .qpsolvers import __version__
from .qpsolvers import available_solvers
from .qpsolvers import cvxopt_solve_qp
from .qpsolvers import cvxpy_solve_qp
from .qpsolvers import dense_solvers
from .qpsolvers import gurobi_solve_qp
from .qpsolvers import mosek_solve_qp
from .qpsolvers import qpoases_solve_qp
from .qpsolvers import quadprog_solve_qp
from .qpsolvers import solve_ls
from .qpsolvers import solve_qp
from .qpsolvers import solve_safer_qp
from .qpsolvers import sparse_solvers

__all__ = [
    "__version__",
    "available_solvers",
    "cvxopt_solve_qp",
    "cvxpy_solve_qp",
    "dense_solvers",
    "gurobi_solve_qp",
    "mosek_solve_qp",
    "qpoases_solve_qp",
    "quadprog_solve_qp",
    "solve_ls",
    "solve_qp",
    "solve_safer_qp",
    "sparse_solvers",
]
