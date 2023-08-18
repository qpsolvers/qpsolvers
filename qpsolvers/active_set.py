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

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ActiveSet:

    """Indices of active constraints for each problem matrix."""

    G_indices: Sequence[int]
    A_indices: Sequence[int]
    lb_indices: Sequence[int]
    ub_indices: Sequence[int]
