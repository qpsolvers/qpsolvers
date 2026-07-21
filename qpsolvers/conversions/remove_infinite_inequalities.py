#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Drop linear inequalities with an infinite right-hand side."""

from typing import Any, Tuple, TypeVar, cast

import numpy as np

from ..exceptions import ProblemError

GType = TypeVar("GType")


def remove_infinite_inequalities(
    G: GType,
    h: np.ndarray,
) -> Tuple[GType, np.ndarray, np.ndarray]:
    r"""Remove linear inequalities disabled by an infinite right-hand side.

    A linear inequality :math:`G_i x \leq h_i` with :math:`h_i = +\infty` is
    always satisfied and can be dropped without changing the problem. This
    function removes such rows so that back-end solvers that don't handle
    infinite values in inequality vectors receive a finite problem. Restore the
    corresponding dual multipliers afterwards with
    :func:`put_infinite_inequalities_back`.

    Parameters
    ----------
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.

    Returns
    -------
    :
        Triple ``(G, h, kept)`` where ``G`` and ``h`` only contain the finite
        inequalities, and ``kept`` is a boolean mask, over the rows of the
        input ``h``, that is ``True`` for kept inequalities and ``False`` for
        the infinite ones that were removed.

    Raises
    ------
    ProblemError :
        If ``h`` contains a :math:`-\infty` or NaN value.
    """
    if np.any(np.isnan(h)):
        raise ProblemError("inequality vector 'h' contains NaN")
    elif np.any(np.isneginf(h)):
        raise ProblemError("inequality vector 'h' contains -infinity")
    kept: np.ndarray = np.isfinite(h)
    if kept.all():
        return G, h, kept
    # G can be dense or sparse matrix supporting boolean row indexing
    G_kept = cast(GType, cast(Any, G)[kept])
    return G_kept, h[kept], kept


def put_infinite_inequalities_back(
    z: np.ndarray,
    kept: np.ndarray,
) -> np.ndarray:
    """Restore multipliers of removed inequalities removed.

    This is the counterpart of :func:`remove_infinite_inequalities`: it maps a
    vector of dual multipliers, computed for the kept (finite) inequalities,
    back to the full set of inequalities. Multipliers of removed inequalities
    are set to zero, as those constraints are inactive by construction.

    Parameters
    ----------
    z :
        Dual multipliers of the kept inequalities.
    kept :
        Boolean mask returned by :func:`remove_infinite_inequalities`.

    Returns
    -------
    :
        Dual multipliers over the full set of inequalities, with zeros at the
        indices of removed inequalities.
    """
    if kept.all():
        return z
    z_full = np.zeros(kept.shape[0], dtype=z.dtype)
    z_full[kept] = z
    return z_full
