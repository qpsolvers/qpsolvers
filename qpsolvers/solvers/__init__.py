#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Import available QP solvers."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from numpy import ndarray
from scipy.sparse import csc_matrix

from ..problem import Problem
from ..solution import Solution

available_solvers: List[str] = []
dense_solvers: List[str] = []
solve_function: Dict[str, Any] = {}
sparse_solvers: List[str] = []

# Clarabel.rs
# ===========

clarabel_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

clarabel_solve_qp: Optional[
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
    from .clarabel_ import clarabel_solve_problem, clarabel_solve_qp

    solve_function["clarabel"] = clarabel_solve_problem
    available_solvers.append("clarabel")
    sparse_solvers.append("clarabel")
except ImportError:
    pass


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


# DAQP
# ========

daqp_solve_qp: Optional[
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

daqp_solve_problem: Optional[
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
    from .daqp_ import daqp_solve_problem, daqp_solve_qp

    solve_function["daqp"] = daqp_solve_problem
    available_solvers.append("daqp")
    dense_solvers.append("daqp")
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


# HPIPM
# =====

hpipm_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            str,
            bool,
        ],
        Solution,
    ]
] = None

hpipm_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
            ndarray,
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            Optional[ndarray],
            str,
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .hpipm_ import hpipm_solve_problem, hpipm_solve_qp

    solve_function["hpipm"] = hpipm_solve_problem
    available_solvers.append("hpipm")
    dense_solvers.append("hpipm")
except ImportError:
    pass

# jaxopt.OSQP
# ==========

jaxopt_osqp_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

jaxopt_osqp_solve_qp: Optional[
    Callable[
        [
            Union[ndarray, csc_matrix],
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
    from .jaxopt_osqp_ import jaxopt_osqp_solve_problem, jaxopt_osqp_solve_qp

    solve_function["jaxopt_osqp"] = jaxopt_osqp_solve_problem
    available_solvers.append("jaxopt_osqp")
    dense_solvers.append("jaxopt_osqp")
except ImportError:
    pass

# KVXOPT
# ======

kvxopt_solve_problem: Optional[
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

kvxopt_solve_qp: Optional[
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
    from .kvxopt_ import kvxopt_solve_problem, kvxopt_solve_qp

    solve_function["kvxopt"] = kvxopt_solve_problem
    available_solvers.append("kvxopt")
    dense_solvers.append("kvxopt")
    sparse_solvers.append("kvxopt")
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


# PIQP
# =======

piqp_solve_qp: Optional[
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

piqp_solve_problem: Optional[
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
    from .piqp_ import piqp_solve_problem, piqp_solve_qp

    solve_function["piqp"] = piqp_solve_problem
    available_solvers.append("piqp")
    dense_solvers.append("piqp")
    sparse_solvers.append("piqp")
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


# QPALM
# =====

qpalm_solve_qp: Optional[
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
        ],
        Optional[ndarray],
    ]
] = None

qpalm_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[Union[ndarray, csc_matrix]],
            bool,
        ],
        Solution,
    ]
] = None

try:
    from .qpalm_ import qpalm_solve_problem, qpalm_solve_qp

    solve_function["qpalm"] = qpalm_solve_problem
    available_solvers.append("qpalm")
    sparse_solvers.append("qpalm")
except ImportError:
    pass

# qpax
# ========

qpax_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
        ],
        Solution,
    ]
] = None

qpax_solve_qp: Optional[
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
    from .qpax_ import qpax_solve_problem, qpax_solve_qp

    solve_function["qpax"] = qpax_solve_problem
    available_solvers.append("qpax")
    dense_solvers.append("qpax")
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


# SIP
# ========

sip_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
            bool,
            bool,
        ],
        Solution,
    ]
] = None

sip_solve_qp: Optional[
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
            bool,
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .sip_ import sip_solve_problem, sip_solve_qp

    solve_function["sip"] = sip_solve_problem
    available_solvers.append("sip")
    sparse_solvers.append("sip")
except ImportError:
    pass


if not available_solvers:
    warnings.warn(
        "no QP solver found on your system, "
        "you can install solvers from PyPI by "
        "``pip install qpsolvers[open_source_solvers]``"
    )


__all__ = [
    "available_solvers",
    "clarabel_solve_qp",
    "cvxopt_solve_qp",
    "daqp_solve_qp",
    "dense_solvers",
    "ecos_solve_qp",
    "gurobi_solve_qp",
    "highs_solve_qp",
    "hpipm_solve_qp",
    "jaxopt_osqp_solve_qp",
    "kvxopt_solve_qp",
    "mosek_solve_qp",
    "osqp_solve_qp",
    "piqp_solve_qp",
    "proxqp_solve_qp",
    "qpalm_solve_qp",
    "qpax_solve_qp",
    "qpoases_solve_qp",
    "qpswift_solve_qp",
    "quadprog_solve_qp",
    "scs_solve_qp",
    "sip_solve_qp",
    "solve_function",
    "sparse_solvers",
]
