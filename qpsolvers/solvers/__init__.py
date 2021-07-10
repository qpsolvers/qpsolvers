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

from typing import Any, Dict

available_solvers = []
dense_solvers = []
solve_function: Dict[str, Any] = {}
sparse_solvers = []


# CVXOPT
# ======

try:
    from .cvxopt_ import cvxopt_solve_qp

    solve_function["cvxopt"] = cvxopt_solve_qp
    available_solvers.append("cvxopt")
    dense_solvers.append("cvxopt")
except ImportError:
    pass


# CVXPY
# =====

try:
    from .cvxpy_ import cvxpy_solve_qp

    solve_function["cvxpy"] = cvxpy_solve_qp
    available_solvers.append("cvxpy")
    sparse_solvers.append("cvxpy")
except ImportError:
    pass


# ECOS
# ====

try:
    from .ecos_ import ecos_solve_qp

    solve_function["ecos"] = ecos_solve_qp
    available_solvers.append("ecos")
    dense_solvers.append("ecos")  # considered dense as it calls cholesky(P)
except ImportError:
    pass


# Gurobi
# ======

try:
    from .gurobi_ import gurobi_solve_qp

    solve_function["gurobi"] = gurobi_solve_qp
    available_solvers.append("gurobi")
    sparse_solvers.append("gurobi")
except ImportError:
    pass


# MOSEK
# =====

try:
    from .mosek_ import mosek_solve_qp

    solve_function["mosek"] = mosek_solve_qp
    available_solvers.append("mosek")
    sparse_solvers.append("mosek")
except ImportError:
    pass


# OSQP
# ====

try:
    from .osqp_ import osqp_solve_qp

    solve_function["osqp"] = osqp_solve_qp
    available_solvers.append("osqp")
    sparse_solvers.append("osqp")
except ImportError:
    pass


# qpOASES
# =======

try:
    from .qpoases_ import qpoases_solve_qp

    solve_function["qpoases"] = qpoases_solve_qp
    available_solvers.append("qpoases")
    dense_solvers.append("qpoases")
except ImportError:
    pass


# quadprog
# ========

try:
    from .quadprog_ import quadprog_solve_qp

    solve_function["quadprog"] = quadprog_solve_qp
    available_solvers.append("quadprog")
    dense_solvers.append("quadprog")
except ImportError:
    pass


# SCS
# ========

try:
    from .scs_ import scs_solve_qp

    solve_function["scs"] = scs_solve_qp
    available_solvers.append("scs")
    dense_solvers.append("scs")
except ImportError:
    pass


__all__ = [
    "available_solvers",
    "cvxopt_solve_qp",
    "cvxpy_solve_qp",
    "dense_solvers",
    "gurobi_solve_qp",
    "mosek_solve_qp",
    "qpoases_solve_qp",
    "quadprog_solve_qp",
    "solve_function",
    "sparse_solvers",
]
