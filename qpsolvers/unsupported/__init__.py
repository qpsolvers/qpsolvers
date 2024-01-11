#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2023 Stéphane Caron and the qpsolvers contributors

"""Import unsupported QP solvers."""

from typing import Any, Callable, Dict, List, Optional

from numpy import ndarray

from ..problem import Problem
from ..solution import Solution

available_solvers: List[str] = []
dense_solvers: List[str] = []
solve_function: Dict[str, Any] = {}
sparse_solvers: List[str] = []


# NPPro
# =====

nppro_solve_problem: Optional[
    Callable[
        [
            Problem,
            Optional[ndarray],
        ],
        Solution,
    ]
] = None

nppro_solve_qp: Optional[
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
        ],
        Optional[ndarray],
    ]
] = None

try:
    from .nppro_ import nppro_solve_problem, nppro_solve_qp

    solve_function["nppro"] = nppro_solve_problem
    available_solvers.append("nppro")
    dense_solvers.append("nppro")
except ImportError:
    pass


__all__ = [
    "available_solvers",
    "nppro_solve_qp",
    "solve_function",
]
