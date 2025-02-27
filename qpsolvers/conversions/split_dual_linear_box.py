#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Convert stacked dual multipliers into linear and box multipliers."""

from typing import Optional, Tuple

import numpy as np


def split_dual_linear_box(
    z_stacked: np.ndarray,
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Separate linear and box multipliers from a stacked dual vector.

    This function assumes linear and box inequalities were combined using
    :func:`qpsolvers.conversions.linear_from_box_inequalities`.

    Parameters
    ----------
    z_stacked :
        Stacked vector of dual multipliers.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.

    Returns
    -------
    :
        Pair :code:`z, z_box` of linear and box multipliers. Both can be empty
        arrays if there is no corresponding constraint.
    """
    z = np.empty((0,), dtype=z_stacked.dtype)
    z_box = np.empty((0,), dtype=z_stacked.dtype)
    if lb is not None and ub is not None:
        n_lb = lb.shape[0]
        n_ub = ub.shape[0]
        n_box = n_lb + n_ub
        z_box = z_stacked[-n_ub:] - z_stacked[-n_box:-n_ub]
        z = z_stacked[:-n_box]
    elif ub is not None:  # lb is None
        n_ub = ub.shape[0]
        z_box = z_stacked[-n_ub:]
        z = z_stacked[:-n_ub]
    elif lb is not None:  # ub is None
        n_lb = lb.shape[0]
        z_box = -z_stacked[-n_lb:]
        z = z_stacked[:-n_lb]
    else:  # lb is None and ub is None
        z = z_stacked
    return z, z_box
