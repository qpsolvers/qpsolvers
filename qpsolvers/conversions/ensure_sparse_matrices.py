#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Make sure problem matrices are sparse."""

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
