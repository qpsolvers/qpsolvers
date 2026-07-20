#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Convert problems from and to standard QP form."""

from .combine_linear_box_inequalities import combine_linear_box_inequalities
from .ensure_sparse_matrices import ensure_sparse_matrices
from .linear_from_box_inequalities import linear_from_box_inequalities
from .remove_infinite_inequalities import (
    put_infinite_inequalities_back,
    remove_infinite_inequalities,
)
from .socp_from_qp import socp_from_qp
from .split_dual_linear_box import split_dual_linear_box

__all__ = [
    "combine_linear_box_inequalities",
    "ensure_sparse_matrices",
    "linear_from_box_inequalities",
    "put_infinite_inequalities_back",
    "remove_infinite_inequalities",
    "socp_from_qp",
    "split_dual_linear_box",
]
