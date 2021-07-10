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

from typing import Any, Callable, Dict, Optional

from numpy import ndarray

from .typing import CvxoptReadyMatrix
from .typing import OsqpReadyMatrix

available_solvers = []
dense_solvers = []
solve_function: Dict[str, Any] = {}
sparse_solvers = []


# CVXOPT
# ======

cvxopt_solve_qp: Optional[
    Callable[
        [
            CvxoptReadyMatrix,            # P
            CvxoptReadyMatrix,            # q
            Optional[CvxoptReadyMatrix],  # G
            Optional[CvxoptReadyMatrix],  # h
            Optional[CvxoptReadyMatrix],  # A
            Optional[CvxoptReadyMatrix],  # b
            Optional[str],                # solver
            Optional[CvxoptReadyMatrix],  # initvals
            bool,                         # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .cvxopt_ import cvxopt_solve_qp

    solve_function["cvxopt"] = cvxopt_solve_qp
    available_solvers.append("cvxopt")
    dense_solvers.append("cvxopt")
except ImportError:
    pass


# CVXPY
# =====

cvxpy_solve_qp: Optional[
    Callable[
        [
            ndarray,            # P
            ndarray,            # q
            Optional[ndarray],  # G
            Optional[ndarray],  # h
            Optional[ndarray],  # A
            Optional[ndarray],  # b
            Optional[ndarray],  # initvals
            Optional[str],      # solver
            bool,               # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .cvxpy_ import cvxpy_solve_qp

    solve_function["cvxpy"] = cvxpy_solve_qp
    available_solvers.append("cvxpy")
    sparse_solvers.append("cvxpy")
except ImportError:
    pass


# ECOS
# ====

ecos_solve_qp: Optional[
    Callable[
        [
            ndarray,            # P
            ndarray,            # q
            Optional[ndarray],  # G
            Optional[ndarray],  # h
            Optional[ndarray],  # A
            Optional[ndarray],  # b
            Optional[ndarray],  # initvals
            bool,               # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .ecos_ import ecos_solve_qp

    solve_function["ecos"] = ecos_solve_qp
    available_solvers.append("ecos")
    dense_solvers.append("ecos")  # considered dense as it calls cholesky(P)
except ImportError:
    pass


# Gurobi
# ======

gurobi_solve_qp: Optional[
    Callable[
        [
            ndarray,            # P
            ndarray,            # q
            Optional[ndarray],  # G
            Optional[ndarray],  # h
            Optional[ndarray],  # A
            Optional[ndarray],  # b
            Optional[ndarray],  # initvals
            bool,               # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .gurobi_ import gurobi_solve_qp

    solve_function["gurobi"] = gurobi_solve_qp
    available_solvers.append("gurobi")
    sparse_solvers.append("gurobi")
except ImportError:
    pass


# MOSEK
# =====

mosek_solve_qp: Optional[
    Callable[
        [
            CvxoptReadyMatrix,            # P
            CvxoptReadyMatrix,            # q
            Optional[CvxoptReadyMatrix],  # G
            Optional[CvxoptReadyMatrix],  # h
            Optional[CvxoptReadyMatrix],  # A
            Optional[CvxoptReadyMatrix],  # b
            Optional[CvxoptReadyMatrix],  # initvals
            bool,                         # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .mosek_ import mosek_solve_qp

    solve_function["mosek"] = mosek_solve_qp
    available_solvers.append("mosek")
    sparse_solvers.append("mosek")
except ImportError:
    mosek_solve_qp = None


# OSQP
# ====

osqp_solve_qp: Optional[
    Callable[
        [
            OsqpReadyMatrix,            # P
            OsqpReadyMatrix,            # q
            Optional[OsqpReadyMatrix],  # G
            Optional[OsqpReadyMatrix],  # h
            Optional[OsqpReadyMatrix],  # A
            Optional[OsqpReadyMatrix],  # b
            Optional[OsqpReadyMatrix],  # initvals
            bool,                       # verbose
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .osqp_ import osqp_solve_qp

    solve_function["osqp"] = osqp_solve_qp
    available_solvers.append("osqp")
    sparse_solvers.append("osqp")
except ImportError:
    osqp_solve_qp = None


# qpOASES
# =======

qpoases_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[str],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

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
    quadprog_solve_qp = None


# SCS
# ========

scs_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[str],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

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
    "osqp_solve_qp",
    "qpoases_solve_qp",
    "quadprog_solve_qp",
    "solve_function",
    "sparse_solvers",
]
