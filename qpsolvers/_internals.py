#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2023 Stéphane Caron and the qpsolvers contributors

"""Internal objects."""

from .solvers import available_solvers as supported_solvers
from .solvers import solve_function as supported_solve
from .unsupported import available_solvers as unsupported_solvers
from .unsupported import solve_function as unsupported_solve

available_solvers = supported_solvers + unsupported_solvers

solve_function = {**supported_solve, **unsupported_solve}
