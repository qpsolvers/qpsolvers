#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Output from a QP solver."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .problem import Problem


@dataclass(frozen=False)
class Solution:
    """Solution returned by a QP solver for a given problem.

    Attributes
    ----------
    extras :
        Other outputs, specific to each solver.

    found :
        True if the solution was found successfully by a solver, False if the
        solver did not find a solution or detected an unfeasible problem,
        ``None`` if no solver was run.

    problem :
        Quadratic program the solution corresponds to.

    obj :
        Value of the primal objective at the solution (``None`` if no solution
        was found).

    x :
        Solution vector for the primal quadratic program (``None`` if no
        solution was found).

    y :
        Dual multipliers for equality constraints (``None`` if no solution was
        found, or if there is no equality constraint). The dimension of
        :math:`y` is equal to the number of equality constraints. The values
        :math:`y_i` can be either positive or negative.

    z :
        Dual multipliers for linear inequality constraints (``None`` if no
        solution was found, or if there is no inequality constraint). The
        dimension of :math:`z` is equal to the number of inequalities. The
        value :math:`z_i` for inequality :math:`i` is always positive.

        - If :math:`z_i > 0`, the inequality is active at the solution:
          :math:`G_i x = h_i`.
        - If :math:`z_i = 0`, the inequality is inactive at the solution:
          :math:`G_i x < h_i`.

    z_box :
        Dual multipliers for box inequality constraints (``None`` if no
        solution was found, or if there is no box inequality). The sign of
        :math:`z_{box,i}` depends on the active bound:

        - If :math:`z_{box,i} < 0`, then the lower bound :math:`lb_i = x_i` is
          active at the solution.
        - If :math:`z_{box,i} = 0`, then neither the lower nor the upper bound
          are active and :math:`lb_i < x_i < ub_i`.
        - If :math:`z_{box,i} > 0`, then the upper bound :math:`x_i = ub_i` is
          active at the solution.
    """

    problem: Problem
    extras: dict = field(default_factory=dict)
    found: Optional[bool] = None
    obj: Optional[float] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    z_box: Optional[np.ndarray] = None

    def is_optimal(self, eps_abs: float) -> bool:
        """Check whether the solution is indeed optimal.

        Parameters
        ----------
        eps_abs :
            Absolute tolerance for the primal residual, dual residual and
            duality gap.

        Notes
        -----
        See for instance [Caron2022]_ for an overview of optimality conditions
        in quadratic programming.
        """
        return (
            self.primal_residual() < eps_abs
            and self.dual_residual() < eps_abs
            and self.duality_gap() < eps_abs
        )

    def primal_residual(self) -> float:
        r"""Compute the primal residual of the solution.

        The primal residual is:

        .. math::

            r_p := \max(\| A x - b \|_\infty, [G x - h]^+,
            [lb - x]^+, [x - ub]^+)

        were :math:`v^- = \min(v, 0)` and :math:`v^+ = \max(v, 0)`.

        Returns
        -------
        :
            Primal residual if it is defined, ``np.inf`` otherwise.

        Notes
        -----
        See for instance [Caron2022]_ for an overview of optimality conditions
        and why this residual will be zero at the optimum.
        """
        _, _, G, h, A, b, lb, ub = self.problem.unpack()
        if not self.found or self.x is None:
            return np.inf
        x = self.x
        return max(
            [
                0.0,
                np.max(G.dot(x) - h) if G is not None else 0.0,
                np.max(np.abs(A.dot(x) - b)) if A is not None else 0.0,
                np.max(lb - x) if lb is not None else 0.0,
                np.max(x - ub) if ub is not None else 0.0,
            ]
        )

    def dual_residual(self) -> float:
        r"""Compute the dual residual of the solution.

        The dual residual is:

        .. math::

            r_d := \| P x + q + A^T y + G^T z + z_{box} \|_\infty

        Returns
        -------
        :
            Dual residual if it is defined, ``np.inf`` otherwise.

        Notes
        -----
        See for instance [Caron2022]_ for an overview of optimality conditions
        and why this residual will be zero at the optimum.
        """
        P, q, G, _, A, _, lb, ub = self.problem.unpack()
        if not self.found or self.x is None:
            return np.inf
        zeros = np.zeros(self.x.shape)
        Px = P.dot(self.x)

        ATy = zeros
        if A is not None:
            if self.y is None:
                return np.inf
            ATy = A.T.dot(self.y)

        GTz = zeros
        if G is not None:
            if self.z is None:
                return np.inf
            GTz = G.T.dot(self.z)

        z_box = zeros
        if lb is not None or ub is not None:
            if self.z_box is None:
                return np.inf
            z_box = self.z_box

        p = np.linalg.norm(Px + q + GTz + ATy + z_box, np.inf)
        return p  # type: ignore

    def duality_gap(self) -> float:
        r"""Compute the duality gap of the solution.

        The duality gap is:

        .. math::

            r_g := | x^T P x + q^T x + b^T y + h^T z +
            lb^T z_{box}^- + ub^T z_{box}^+ |

        were :math:`v^- = \min(v, 0)` and :math:`v^+ = \max(v, 0)`.

        Returns
        -------
        :
            Duality gap if it is defined, ``np.inf`` otherwise.

        Notes
        -----
        See for instance [Caron2022]_ for an overview of optimality conditions
        and why this gap will be zero at the optimum.
        """
        P, q, _, h, _, b, lb, ub = self.problem.unpack()
        if not self.found or self.x is None:
            return np.inf
        xPx = self.x.T.dot(P.dot(self.x))
        qx = q.dot(self.x)

        hz = 0.0
        if h is not None:
            if self.z is None:
                return np.inf
            hz = h.dot(self.z)

        by = 0.0
        if b is not None:
            if self.y is None:
                return np.inf
            by = b.dot(self.y)

        lb_z_box = 0.0
        ub_z_box = 0.0
        if self.z_box is not None:
            if lb is not None:
                finite = np.asarray(lb != -np.inf).nonzero()
                z_box_neg = np.minimum(self.z_box, 0.0)
                lb_z_box = lb[finite].dot(z_box_neg[finite])
            if ub is not None:
                finite = np.asarray(ub != np.inf).nonzero()
                z_box_pos = np.maximum(self.z_box, 0.0)
                ub_z_box = ub[finite].dot(z_box_pos[finite])
        return abs(xPx + qx + hz + by + lb_z_box + ub_z_box)
