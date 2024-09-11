#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023 Inria

"""Active set: indices of inequality constraints saturated at the optimum."""

from dataclasses import dataclass
from typing import Optional, Sequence


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
        G_indices: Optional[Sequence[int]] = None,
        lb_indices: Optional[Sequence[int]] = None,
        ub_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.G_indices = list(G_indices) if G_indices is not None else []
        self.lb_indices = list(lb_indices) if lb_indices is not None else []
        self.ub_indices = list(ub_indices) if ub_indices is not None else []
