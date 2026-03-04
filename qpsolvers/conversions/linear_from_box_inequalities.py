#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Functions to convert vector bounds into linear inequality constraints."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as spa

from ..exceptions import ProblemError


def concatenate_bound(
    G: Optional[Union[np.ndarray, spa.csc_matrix, spa.dia_matrix]],
    h: Optional[np.ndarray],
    b: np.ndarray,
    sign: float,
    use_sparse: bool,
) -> Tuple[Optional[Union[np.ndarray, spa.csc_matrix]], Optional[np.ndarray]]:
    """Append bound constraint vectors to inequality constraints.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    b :
        Bound constraint vector.
    sign :
        Sign factor: -1.0 for a lower and +1.0 for an upper bound.
    use_sparse :
        Use sparse matrices if true, dense matrices otherwise.

    Returns
    -------
    G :
        Updated linear inequality matrix.
    h :
        Updated linear inequality vector.
    """
    n = len(b)  # == number of optimization variables
    if G is None or h is None:
        G = sign * (spa.eye(n, format="csc") if use_sparse else np.eye(n))
        h = sign * b
        return (G, h)

    h = np.concatenate((h, sign * b))
    if isinstance(G, np.ndarray):
        dense_G: np.ndarray = np.concatenate((G, sign * np.eye(n)), 0)
        return (dense_G, h)
    elif isinstance(G, (spa.csc_matrix, spa.dia_matrix)):
        sparse_G: spa.csc_matrix = spa.vstack(
            [G, sign * spa.eye(n)], format="csc"
        )
        return (sparse_G, h)
    else:  # G is not an instance of a type we know
        name = type(G).__name__
        raise ProblemError(f"invalid type '{name}' for inequality matrix G")


def linear_from_box_inequalities(
    G: Optional[Union[np.ndarray, spa.csc_matrix, spa.dia_matrix]],
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
    G :
        Updated linear inequality matrix.
    h :
        Updated linear inequality vector.
    """
    if lb is not None:
        G, h = concatenate_bound(G, h, lb, -1.0, use_sparse)
    if ub is not None:
        G, h = concatenate_bound(G, h, ub, +1.0, use_sparse)
    return (G, h)
