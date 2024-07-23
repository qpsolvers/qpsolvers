#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2023 Stéphane Caron and the qpsolvers contributors

"""Combine linear and box inequalities into double-sided linear format."""

import numpy as np
import scipy.sparse as spa

from ..exceptions import ProblemError


def combine_linear_box_inequalities(G, h, lb, ub, n: int, use_csc: bool):
    r"""Combine linear and box inequalities into double-sided linear format.

    Input format:

    .. math::

        \begin{split}\begin{array}{ll}
            G x & \leq h \\
            lb & \leq x \leq ub
        \end{array}\end{split}

    Output format:

    .. math::

        l \leq C \leq u

    Parameters
    ----------
    G :
        Linear inequality constraint matrix. Must be two-dimensional.
    h :
        Linear inequality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    n :
        Number of optimization variables.
    use_csc :
        If ``True``, use sparse rather than dense matrices.

    Returns
    -------
    :
        Linear inequality matrix :math:`C` and vectors :math:`u`, :math:`l`.
        The two vector will contain :math:`\pm\infty` values on coordinates
        where there is no corresponding constraint.

    Raises
    ------
    ProblemError
        If the inequality matrix and vector are not consistent.
    """
    if lb is None and ub is None:
        C_out = G
        u_out = h
        l_out = np.full(h.shape, -np.inf) if h is not None else None
    elif G is None:
        # lb is not None or ub is not None:
        C_out = spa.eye(n, format="csc") if use_csc else np.eye(n)
        u_out = ub if ub is not None else np.full(n, +np.inf)
        l_out = lb if lb is not None else np.full(n, -np.inf)
    elif h is not None:
        # G is not None and h is not None and not (lb is None and ub is None)
        C_out = (
            spa.vstack((G, spa.eye(n)), format="csc")
            if use_csc
            else np.vstack((G, np.eye(n)))
        )
        ub = ub if ub is not None else np.full(G.shape[1], +np.inf)
        lb = lb if lb is not None else np.full(G.shape[1], -np.inf)
        l_out = np.hstack((np.full(h.shape, -np.inf), lb))
        u_out = np.hstack((h, ub))
    else:  # G is not None and h is None
        raise ProblemError("Inconsistent inequalities: G is set but h is None")
    return C_out, u_out, l_out
