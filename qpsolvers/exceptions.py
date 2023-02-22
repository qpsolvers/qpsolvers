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

"""
Exceptions from qpsolvers.

We catch all solver exceptions and re-throw them in a qpsolvers-owned exception
to avoid abstraction leakage. See this `design decision
<https://github.com/getparthenon/parthenon/wiki/Design-Decision:-Throw-Custom-Exceptions>`__
for more details on the rationale behind this choice.
"""


class QPError(Exception):
    """Base class for qpsolvers exceptions."""


class NoSolverSelected(QPError):
    """Exception raised when the `solver` keyword argument is not set."""


class ParamError(QPError):
    """Exception raised when solver parameters are incorrect."""


class ProblemError(QPError):
    """Exception raised when a quadratic program is malformed."""


class SolverNotFound(QPError):
    """Exception raised when a requested solver is not found."""


class SolverError(QPError):
    """Exception raised when a solver failed."""
