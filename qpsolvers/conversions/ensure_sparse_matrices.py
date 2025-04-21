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

from ..warnings import SparseConversionWarning


def __warn_about_sparse_conversion(matrix_name: str, solver_name: str) -> None:
    """Warn about conversion from dense to sparse matrix.

    Parameters
    ----------
    matrix_name :
        Name of matrix being converted from dense to sparse.
    solver_name :
        Name of the QP solver matrices will be passed to.
    """
    warnings.warn(
        f"Converted matrix '{matrix_name}' of your problem to "
        f"scipy.sparse.csc_matrix to pass it to solver '{solver_name}'; "
        f"for best performance, build your matrix as a csc_matrix directly.",
        category=SparseConversionWarning,
    )


def ensure_sparse_matrices(
    solver_name: str,
    P: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]],
    A: Optional[Union[np.ndarray, spa.csc_matrix]],
) -> Tuple[spa.csc_matrix, Optional[spa.csc_matrix], Optional[spa.csc_matrix]]:
    """
    Make sure problem matrices are sparse.

    Parameters
    ----------
    solver_name :
        Name of the QP solver matrices will be passed to.
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
        __warn_about_sparse_conversion("P", solver_name)
        P = spa.csc_matrix(P)
    if isinstance(G, np.ndarray):
        __warn_about_sparse_conversion("G", solver_name)
        G = spa.csc_matrix(G)
    if isinstance(A, np.ndarray):
        __warn_about_sparse_conversion("A", solver_name)
        A = spa.csc_matrix(A)
    return P, G, A
