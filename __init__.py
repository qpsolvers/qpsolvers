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
This file allows the whole repository to act as a Python module when e.g.
included as a git submodule inside a project. It incurs redundant bookkeeping
but some users find it useful. (See 4b128db662cfedbe366b8387a59cd18e36c643a3
and comments related to this commit on GitHub.)
"""

from .qpsolvers import (
    Problem,
    Solution,
    __version__,
    available_solvers,
    cvxopt_solve_qp,
    dense_solvers,
    ecos_solve_qp,
    gurobi_solve_qp,
    highs_solve_qp,
    mosek_solve_qp,
    osqp_solve_qp,
    print_matrix_vector,
    proxqp_solve_qp,
    qpoases_solve_qp,
    qpswift_solve_qp,
    quadprog_solve_qp,
    scs_solve_qp,
    solve_ls,
    solve_qp,
    solve_safer_qp,
    sparse_solvers,
)

__all__ = [
    "Problem",
    "Solution",
    "__version__",
    "available_solvers",
    "cvxopt_solve_qp",
    "dense_solvers",
    "ecos_solve_qp",
    "gurobi_solve_qp",
    "highs_solve_qp",
    "mosek_solve_qp",
    "osqp_solve_qp",
    "print_matrix_vector",
    "proxqp_solve_qp",
    "qpoases_solve_qp",
    "qpswift_solve_qp",
    "quadprog_solve_qp",
    "scs_solve_qp",
    "solve_ls",
    "solve_qp",
    "solve_safer_qp",
    "sparse_solvers",
]
