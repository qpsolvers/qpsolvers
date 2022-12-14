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

"""Types for solve_qp function arguments."""

import warnings


def warn_about_sparse_conversion(matrix_name: str) -> None:
    """
    Warn about conversion from dense to sparse matrix.

    Parameters
    ----------
    matrix_name :
        Name of matrix being converted from dense to sparse.
    """
    warnings.warn(
        f"Converted {matrix_name} to scipy.sparse.csc.csc_matrix\n"
        f"For best performance, build {matrix_name} as a "
        "scipy.sparse.csc_matrix rather than as a numpy.ndarray"
    )


__all__ = [
    "warn_about_sparse_conversion",
]
