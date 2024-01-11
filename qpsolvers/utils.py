#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Utility functions."""

from typing import Union

import numpy as np
import scipy.sparse as spa


def print_matrix_vector(
    A: Union[np.ndarray, spa.csc_matrix],
    A_label: str,
    b: np.ndarray,
    b_label: str,
    column_width: int = 24,
) -> None:
    """Print a matrix and vector side by side to the terminal.

    Parameters
    ----------
    A :
        Union[np.ndarray, spa.csc_matrix] to print.
    A_label :
        Label for A.
    b :
        np.ndarray to print.
    b_label :
        Label for b.
    column_width :
        Number of characters for the matrix and vector text columns.
    """
    if isinstance(A, np.ndarray) and A.ndim == 1:
        A = A.reshape((1, A.shape[0]))
    if isinstance(A, spa.csc_matrix):
        A = A.toarray()
    if A.shape[0] == b.shape[0]:
        A_string = f"{A_label} =\n{A}"
        b_string = f"{b_label} =\n{b.reshape((A.shape[0], 1))}"
    elif A.shape[0] > b.shape[0]:
        m = b.shape[0]
        A_string = f"{A_label} =\n{A[:m]}"
        b_string = f"{b_label} =\n{b.reshape(m, 1)}"
        A_string += f"\n{A[m:]}"
        b_string += "\n " * (A.shape[0] - m)
    else:  # A.shape[0] < b.shape[0]
        n = A.shape[0]
        k = b.shape[0] - n
        A_string = f"{A_label} =\n{A}"
        b_string = f"{b_label} =\n{b[:n].reshape(n, 1)}"
        A_string += "\n " * k
        b_string += f"\n{b[n:].reshape(k, 1)}"
    A_lines = A_string.splitlines()
    b_lines = b_string.splitlines()
    for i, A_line in enumerate(A_lines):
        print(A_line.ljust(column_width) + b_lines[i].ljust(column_width))
