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

"""Convert problems from and to standard QP form."""

from .linear_from_box_inequalities import linear_from_box_inequalities
from .socp_from_qp import socp_from_qp
from .split_dual_linear_box import split_dual_linear_box
from .warnings import warn_about_sparse_conversion

__all__ = [
    "linear_from_box_inequalities",
    "socp_from_qp",
    "split_dual_linear_box",
    "warn_about_sparse_conversion",
]
