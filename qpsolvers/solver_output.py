#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 The qpsolvers contributors.
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
Type for extended QP solver outputs going beyond the primal solution.
"""

from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class QPOutput:
    """
    This is the generic Solver output object. This is the general extra infor return object for solvers. It contains\\
    all of the information you would need for the optimization solution including, optimal value, optimal solution, the \\
    active set, the value of the slack variables and the largange multipliers associated with every constraint.
    Parameters
    ----------
    obj: optimal objective \n
    sol: x*, numpy array \n
    Optional Parameters -> None or numpy.ndarray type
    slack: the slacks associated with every constraint \n
    equality_indices: the active set of the solution, including strongly and weakly active constraints \n
    dual: the lagrange multipliers associated with the problem\n
    """
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]

    def __eq__(self, other):
        if not isinstance(other, QPOutput):
            return NotImplemented

        return numpy.allclose(self.slack, other.slack) and numpy.allclose(self.active_set,
                                                                          other.active_set) and numpy.allclose(
            self.dual, other.dual) and numpy.allclose(self.sol, other.sol) and numpy.allclose(self.obj, other.obj)
