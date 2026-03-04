#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Functions to convert vector bounds into linear inequality constraints."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as spa


def concatenate_bound_dense(
    G: Optional[np.ndarray],
    h: Optional[np.ndarray],
    b: np.ndarray,
    sign: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Append bound constraint vectors to dense inequality constraints.

    Parameters
    ----------
    G :
        Dense linear inequality matrix, or None.
    h :
        Linear inequality vector, or None.
    b :
        Bound constraint vector.
    sign :
        Sign factor: -1.0 for a lower and +1.0 for an upper bound.

    Returns
    -------
    G : numpy.ndarray
        Updated dense linear inequality matrix.
    h : numpy.ndarray
        Updated linear inequality vector.
    """
    n = len(b)
    if G is None or h is None:
        return sign * np.eye(n), sign * b
    return (
        np.concatenate((G, sign * np.eye(n)), 0),
        np.concatenate((h, sign * b)),
    )


def concatenate_bound_sparse(
    G: Optional[spa.csc_matrix],
    h: Optional[np.ndarray],
    b: np.ndarray,
    sign: float,
) -> Tuple[spa.csc_matrix, np.ndarray]:
    """Append bound constraint vectors to sparse inequality constraints.

    Parameters
    ----------
    G :
        Sparse linear inequality matrix, or None.
    h :
        Linear inequality vector, or None.
    b :
        Bound constraint vector.
    sign :
        Sign factor: -1.0 for a lower and +1.0 for an upper bound.

    Returns
    -------
    G : scipy.sparse.csc_matrix
        Updated sparse linear inequality matrix.
    h : numpy.ndarray
        Updated linear inequality vector.
    """
    n = len(b)
    if G is None or h is None:
        return sign * spa.eye(n, format="csc"), sign * b
    return (
        spa.vstack([G, sign * spa.eye(n)], format="csc"),
        np.concatenate((h, sign * b)),
    )


def sparse_linear_from_box_inequalities(
    G: Optional[spa.csc_matrix],
    h: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[spa.csc_matrix], Optional[np.ndarray]]:
    """Append lower or upper bound vectors to sparse inequality constraints.

    Parameters
    ----------
    G :
        Sparse linear inequality matrix, or None.
    h :
        Linear inequality vector, or None.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.

    Returns
    -------
    G : scipy.sparse.csc_matrix or None
        Updated sparse linear inequality matrix.
    h : numpy.ndarray or None
        Updated linear inequality vector.
    """
    if lb is not None:
        G, h = concatenate_bound_sparse(G, h, lb, -1.0)
    if ub is not None:
        G, h = concatenate_bound_sparse(G, h, ub, +1.0)
    return (G, h)


def linear_from_box_inequalities(
    G: Optional[Union[np.ndarray, spa.csc_matrix]],
    h: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
    use_sparse: bool,
) -> Tuple[Optional[Union[np.ndarray, spa.csc_matrix]], Optional[np.ndarray]]:
    """Append lower or upper bound vectors to inequality constraints.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    use_sparse :
        Use sparse matrices if true, dense matrices otherwise.

    Returns
    -------
    G : np.ndarray, spa.csc_matrix or None
        Updated linear inequality matrix.
    h : np.ndarray or None
        Updated linear inequality vector.
    """
    if use_sparse:
        sparse_G: Optional[spa.csc_matrix] = (
            G if G is None or isinstance(G, spa.csc_matrix) else spa.csc_matrix(G)
        )
        if lb is not None:
            sparse_G, h = concatenate_bound_sparse(sparse_G, h, lb, -1.0)
        if ub is not None:
            sparse_G, h = concatenate_bound_sparse(sparse_G, h, ub, +1.0)
        return (sparse_G, h)
    dense_G: Optional[np.ndarray] = (
        G if G is None or isinstance(G, np.ndarray) else G.toarray()
    )
    if lb is not None:
        dense_G, h = concatenate_bound_dense(dense_G, h, lb, -1.0)
    if ub is not None:
        dense_G, h = concatenate_bound_dense(dense_G, h, ub, +1.0)
    return (dense_G, h)
