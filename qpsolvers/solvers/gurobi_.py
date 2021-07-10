#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stephane Caron <stephane.caron@normalesup.org>
# Copyright (C) 2021 Dustin Kenefake
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

"""Solver interface for Gurobi"""

from typing import Optional
from warnings import warn

from gurobipy import GRB, Model
from numpy import array, ndarray


def gurobi_solve_qp(
    P, q, G=None, h=None, A=None, b=None, initvals=None, verbose: bool = False
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = h
        \\end{array}\\end{split}

    using `Gurobi <http://www.gurobi.com/>`_.

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    initvals : array, shape=(n,), optional
        Warm-start guess vector (not used).
    verbose : bool, optional
        Set to `True` to print out extra information.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        warn("Gurobi: warm-start values given but they will be ignored")
    model = Model()
    if not verbose:  # optionally turn off solver output
        model.setParam("OutputFlag", 0)
    num_vars = P.shape[0]
    x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    if A is not None:  # include equality constraints
        model.addMConstr(A, x, GRB.EQUAL, b)
    if G is not None:  # include inequality constraints
        model.addMConstr(G, x, GRB.LESS_EQUAL, h)
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, sense=GRB.MINIMIZE)
    model.optimize()
    status = model.status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None
    return array(x.X)
