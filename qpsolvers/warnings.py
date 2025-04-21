#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2025 Inria

"""Warnings from qpsolvers."""


class QPWarning(UserWarning):
    """Base class for qpsolvers warnings."""


class SparseConversionWarning(QPWarning):
    """Warning issued when converting NumPy arrays to SciPy sparse matrices."""
