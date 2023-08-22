#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""Active set: indices of inequality constraints saturated at the optimum."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ActiveSet:
    """Indices of active inequality constraints.

    Attributes
    ----------
    G_indices :
        Indices of active linear inequality constraints.
    lb_indices :
        Indices of active lower-bound inequality constraints.
    ub_indices :
        Indices of active upper-bound inequality constraints.
    """

    G_indices: Sequence[int]
    lb_indices: Sequence[int]
    ub_indices: Sequence[int]

    def __init__(
        self,
        G_indices: Sequence[int] = [],
        lb_indices: Sequence[int] = [],
        ub_indices: Sequence[int] = [],
    ) -> None:
        self.G_indices = list(G_indices)
        self.lb_indices = list(lb_indices)
        self.ub_indices = list(ub_indices)
