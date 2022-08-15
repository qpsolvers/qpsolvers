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
Solver interface for `ProxQP <https://github.com/qpSWIFT/qpSWIFT>`__.

ProxQP is the QP solver from ProxSuite, a collection of open-source solvers
rooted in revisited primal-dual proximal algorithms. If you are using ProxQP in
your work, consider citing the corresponding paper: `PROX-QP: Yet another
Quadratic Programming Solver for Robotics and beyond
<https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/>`__.
"""

from typing import Optional

import numpy as np
import proxsuite

from .conversions import linear_from_box_inequalities


def proxsuite_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b                    \\\\
            & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `ProxQP <https://github.com/Simple-Robotics/proxsuite>`__.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix. Together with :math:`A` and :math:`G`,
        it should satisfy :math:`\\mathrm{rank}([P\\ A^T\\ G^T]) = n`, see the
        rank assumptions below.
    q :
        Quadratic-cost vector.
    G :
        Linear inequality constraint matrix. Together with :math:`P` and
        :math:`A`, it should satisfy :math:`\\mathrm{rank}([P\\ A^T\\ G^T]) =
        n`, see the rank assumptions below.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix. It needs to be full row rank, and
        together with :math:`P` and :math:`G` satisfy
        :math:`\\mathrm{rank}([P\\ A^T\\ G^T]) = n`. See the rank assumptions
        below.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    All other keyword arguments are forwarded as options to ProxQP. For
    instance, you can call ``proxqp_solve_qp(P, q, G, h, TODO(scaron): ...)``.
    See the solver documentation for details.
    """
    if initvals is not None:
        print("ProxQP: note that warm-start values ignored by wrapper")
    return None
