#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2023 Stéphane Caron and the qpsolvers contributors.
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

"""Import unsupported QP solvers."""

from typing import Any, Callable, Dict, Optional, Union

from numpy import ndarray
from scipy.sparse import csc_matrix

from ..problem import Problem
from ..solution import Solution

available_solvers = []
dense_solvers = []
solve_function: Dict[str, Any] = {}
sparse_solvers = []


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
    "mosek_solve_qp",
    "nppro_solve_qp",
    "solve_function",
]
