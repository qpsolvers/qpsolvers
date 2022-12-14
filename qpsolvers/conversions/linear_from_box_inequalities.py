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

"""Functions to convert vector bounds into linear inequality constraints."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as spa


def concatenate_bound(
    G: Optional[Union[np.ndarray, spa.csc_matrix]],
    h: Optional[np.ndarray],
    b: np.ndarray,
    sign: float,
) -> Tuple[Optional[Union[np.ndarray, spa.csc_matrix]], Optional[np.ndarray]]:
    """
    Append bound constraint vectors to inequality constraints.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    b:
        Bound constraint vector.
    sign:
        Sign factor: -1.0 for a lower and +1.0 for an upper bound.

    Returns
    -------
    G : numpy.ndarray, scipy.sparse.csc_matrix, or None
        Updated linear inequality matrix.
    h : numpy.ndarray or None
        Updated linear inequality vector.
    """
    n = len(b)  # == number of optimization variables
    if G is None or h is None:
        G = sign * np.eye(n)
        h = sign * b
    else:  # G is not None and h is not None
        if isinstance(G, np.ndarray):
            G = np.concatenate((G, sign * np.eye(n)), 0)
        elif isinstance(G, (spa.csc_matrix, spa.dia_matrix)):
            G = spa.vstack([G, sign * spa.eye(n)], format="csc")
        else:  # G is not an instance of a type we know
            name = type(G).__name__
            raise TypeError(f"invalid type '{name}' for inequality matrix G")
        h = np.concatenate((h, sign * b))
    return (G, h)


def linear_from_box_inequalities(
    G: Optional[Union[np.ndarray, spa.csc_matrix]],
    h: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[Union[np.ndarray, spa.csc_matrix]], Optional[np.ndarray]]:
    """
    Append lower or upper bound constraint vectors to inequality constraints.

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

    Returns
    -------
    G : np.ndarray, spa.csc_matrix or None
        Updated linear inequality matrix.
    h : np.ndarray or None
        Updated linear inequality vector.
    """
    if lb is not None:
        G, h = concatenate_bound(G, h, lb, -1.0)
    if ub is not None:
        G, h = concatenate_bound(G, h, ub, +1.0)
    return (G, h)
