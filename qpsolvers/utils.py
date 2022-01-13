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

"""Utility functions"""

from numpy import ndarray
from scipy.sparse import csc_matrix

from .typing import Matrix, Vector


def print_matrix_vector(
    A: Matrix, A_label: str, b: Vector, b_label: str, column_width: int = 24
) -> None:
    """
    Print a matrix and vector side by side to the terminal.

    Parameters
    ----------
    A :
        Matrix to print.
    A_label :
        Label for A.
    b :
        Vector to print.
    b_label :
        Label for b.
    column_width :
        Number of characters for the matrix and vector text columns.
    """
    if isinstance(A, ndarray) and A.ndim == 1:
        A = A.reshape((1, A.shape[0]))
    if isinstance(A, csc_matrix):
        A = A.toarray()
    A_string = f"{A_label} =\n{A}"
    b_string = f"{b_label} =\n{b.reshape((A.shape[0], 1))}"
    A_lines = A_string.splitlines()
    b_lines = b_string.splitlines()
    for i in range(len(A_lines)):
        print(A_lines[i].ljust(column_width) + b_lines[i].ljust(column_width))
