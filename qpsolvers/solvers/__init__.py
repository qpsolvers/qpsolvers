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

"""Import available QP solvers."""

from typing import Any, Callable, Dict, Optional, Union

from numpy import ndarray
from scipy.sparse import csc_matrix

from ..problem import Problem
from ..solution import Solution

available_solvers = []
dense_solvers = []
solve_function: Dict[str, Any] = {}
sparse_solvers = []

# CVXOPT
# ======

cvxopt_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[str],
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

cvxopt_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[Union[ndarray, csc_matrix]],
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
    from .cvxopt_ import cvxopt_solve_problem, cvxopt_solve_qp

    solve_function["cvxopt"] = cvxopt_solve_problem
    available_solvers.append("cvxopt")
    dense_solvers.append("cvxopt")
    sparse_solvers.append("cvxopt")
except ImportError:
    pass


# ECOS
# ====

ecos_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

ecos_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .ecos_ import ecos_solve_problem, ecos_solve_qp

    solve_function["ecos"] = ecos_solve_problem
    available_solvers.append("ecos")
    dense_solvers.append("ecos")  # considered dense as it calls cholesky(P)
except ImportError:
    pass


# Gurobi
# ======

gurobi_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

gurobi_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .gurobi_ import gurobi_solve_problem, gurobi_solve_qp

    solve_function["gurobi"] = gurobi_solve_problem
    available_solvers.append("gurobi")
    sparse_solvers.append("gurobi")
except ImportError:
    pass


# HiGHS
# =====

highs_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

highs_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .highs_ import highs_solve_problem, highs_solve_qp

    solve_function["highs"] = highs_solve_problem
    available_solvers.append("highs")
    sparse_solvers.append("highs")
except ImportError:
    pass


# MOSEK
# =====

mosek_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

mosek_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .mosek_ import mosek_solve_problem, mosek_solve_qp

    solve_function["mosek"] = mosek_solve_problem
    available_solvers.append("mosek")
    sparse_solvers.append("mosek")
except ImportError:
    pass


# OSQP
# ====

osqp_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

osqp_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .osqp_ import osqp_solve_problem, osqp_solve_qp

    solve_function["osqp"] = osqp_solve_problem
    available_solvers.append("osqp")
    sparse_solvers.append("osqp")
except ImportError:
    pass


# ProxQP
# =======

proxqp_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            Union[ndarray, csc_matrix],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            Optional[Union[ndarray, csc_matrix]],
            bool,
            Optional[str],
        ],
        Optional[ndarray],
    ]
] = None

proxqp_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[Union[ndarray, csc_matrix]],
            bool,
            Optional[str],
        ],
        Solution,
    ]
] = None

try:
    from .proxqp_ import proxqp_solve_problem, proxqp_solve_qp

    solve_function["proxqp"] = proxqp_solve_problem
    available_solvers.append("proxqp")
    dense_solvers.append("proxqp")
    sparse_solvers.append("proxqp")
except ImportError:
    pass


# qpOASES
# =======

qpoases_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
            int,
            Optional[float],
        ],
        Solution,
    ]
] = None

qpoases_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
            int,
            Optional[float],
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .qpoases_ import qpoases_solve_problem, qpoases_solve_qp

    solve_function["qpoases"] = qpoases_solve_problem
    available_solvers.append("qpoases")
    dense_solvers.append("qpoases")
except ImportError:
    pass


# qpSWIFT
# =======

qpswift_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

qpswift_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .qpswift_ import qpswift_solve_problem, qpswift_solve_qp

    solve_function["qpswift"] = qpswift_solve_problem
    available_solvers.append("qpswift")
    dense_solvers.append("qpswift")
except ImportError:
    pass


# quadprog
# ========

quadprog_solve_qp: Optional[
    Callable[
        [
            ndarray,
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

quadprog_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

try:
    from .quadprog_ import quadprog_solve_problem, quadprog_solve_qp

    solve_function["quadprog"] = quadprog_solve_problem
    available_solvers.append("quadprog")
    dense_solvers.append("quadprog")
except ImportError:
    pass


# SCS
# ========

scs_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

scs_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[Union[ndarray, csc_matrix]],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .scs_ import scs_solve_problem, scs_solve_qp

    solve_function["scs"] = scs_solve_problem
    available_solvers.append("scs")
    sparse_solvers.append("scs")
except ImportError:
    pass


if not available_solvers:
    raise ImportError(
        "no QP solver found, you can install some by running:\n\n"
        "\tpip install qpsolvers[open_source_solvers]\n"
    )


__all__ = [
    "available_solvers",
    "cvxopt_solve_qp",
    "dense_solvers",
    "ecos_solve_qp",
    "gurobi_solve_qp",
    "highs_solve_qp",
    "mosek_solve_qp",
    "osqp_solve_qp",
    "proxqp_solve_qp",
    "qpoases_solve_qp",
    "qpswift_solve_qp",
    "quadprog_solve_qp",
    "scs_solve_qp",
    "solve_function",
    "sparse_solvers",
]
