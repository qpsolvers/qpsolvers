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

"""Model for a quadratic program."""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as spa


def __warn_about_sparse_conversion(matrix_name: str) -> None:
    """Warn about conversion from dense to sparse matrix.

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


def ensure_sparse_matrices(
    P: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]],
    A: Optional[Union[np.ndarray, spa.csc_matrix]],
) -> Tuple[spa.csc_matrix, Optional[spa.csc_matrix], Optional[spa.csc_matrix]]:
    """
    Make sure problem matrices are sparse.

    Parameters
    ----------
    P :
        Cost matrix.
    G :
        Inequality constraint matrix, if any.
    A :
        Equality constraint matrix, if any.

    Returns
    -------
    :
        Tuple of all three matrices as sparse matrices.
    """
    if isinstance(P, np.ndarray):
        __warn_about_sparse_conversion("P")
        P = spa.csc_matrix(P)
    if isinstance(G, np.ndarray):
        __warn_about_sparse_conversion("G")
        G = spa.csc_matrix(G)
    if isinstance(A, np.ndarray):
        __warn_about_sparse_conversion("A")
        A = spa.csc_matrix(A)
    return P, G, A
