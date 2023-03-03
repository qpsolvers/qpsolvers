#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2023 Stéphane Caron and the qpsolvers contributors.
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

"""Internal objects."""

import warnings

from .solvers import available_solvers as supported_solvers
from .solvers import solve_function as supported_solve
from .unsupported import available_solvers as unsupported_solvers
from .unsupported import solve_function as unsupported_solve

available_solvers = supported_solvers + unsupported_solvers
for solver in unsupported_solvers:
    warnings.warn(f'QP solver "{solver}" is available but unsupported')

solve_function = {**supported_solve, **unsupported_solve}
