#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Quadratic programming solvers in Python with a unified API."""

from ._internals import available_solvers
from .active_set import ActiveSet
from .exceptions import (
    NoSolverSelected,
    ParamError,
    ProblemError,
    QPError,
    SolverError,
    SolverNotFound,
)
from .problem import Problem
from .solution import Solution
from .solve_ls import solve_ls
from .solve_qp import solve_problem, solve_qp
from .solve_unconstrained import solve_unconstrained
from .solvers import (
    cvxopt_solve_qp,
    daqp_solve_qp,
    dense_solvers,
    ecos_solve_qp,
    gurobi_solve_qp,
    highs_solve_qp,
    hpipm_solve_qp,
    kvxopt_solve_qp,
    mosek_solve_qp,
    osqp_solve_qp,
    piqp_solve_qp,
    proxqp_solve_qp,
    qpalm_solve_qp,
    qpoases_solve_qp,
    qpswift_solve_qp,
    quadprog_solve_qp,
    scs_solve_qp,
    sip_solve_qp,
    sparse_solvers,
)
from .unsupported import nppro_solve_qp
from .utils import print_matrix_vector

__version__ = "4.7.1"

__all__ = [
    "ActiveSet",
    "NoSolverSelected",
    "ParamError",
    "Problem",
    "ProblemError",
    "QPError",
    "Solution",
    "SolverError",
    "SolverNotFound",
    "__version__",
    "available_solvers",
    "cvxopt_solve_qp",
    "daqp_solve_qp",
    "dense_solvers",
    "ecos_solve_qp",
    "gurobi_solve_qp",
    "highs_solve_qp",
    "hpipm_solve_qp",
    "kvxopt_solve_qp",
    "mosek_solve_qp",
    "nppro_solve_qp",
    "osqp_solve_qp",
    "print_matrix_vector",
    "piqp_solve_qp",
    "proxqp_solve_qp",
    "qpalm_solve_qp",
    "qpoases_solve_qp",
    "qpswift_solve_qp",
    "quadprog_solve_qp",
    "scs_solve_qp",
    "sip_solve_qp",
    "solve_ls",
    "solve_problem",
    "solve_qp",
    "solve_unconstrained",
    "sparse_solvers",
]
