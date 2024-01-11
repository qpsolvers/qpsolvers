#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Convert problems from and to standard QP form."""

from .combine_linear_box_inequalities import combine_linear_box_inequalities
from .ensure_sparse_matrices import ensure_sparse_matrices
from .linear_from_box_inequalities import linear_from_box_inequalities
from .socp_from_qp import socp_from_qp
from .split_dual_linear_box import split_dual_linear_box

__all__ = [
    "combine_linear_box_inequalities",
    "linear_from_box_inequalities",
    "socp_from_qp",
    "split_dual_linear_box",
    "ensure_sparse_matrices",
]
