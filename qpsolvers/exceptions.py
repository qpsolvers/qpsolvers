#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

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
