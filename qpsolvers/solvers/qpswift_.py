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
Solver interface for `qpSWIFT <https://github.com/qpSWIFT/qpSWIFT>`__.

qpSWIFT is a light-weight sparse Quadratic Programming solver targeted for
embedded and robotic applications. It employs Primal-Dual Interior Point method
with Mehrotra Predictor corrector step and Nesterov Todd scaling. For solving
the linear system of equations, sparse LDL' factorization is used along with
approximate minimum degree heuristic to minimize fill-in of the factorizations.

If you use qpSWIFT in your research, consider citing the corresponding paper:
`qpSWIFT: A Real-Time Sparse Quadratic Program Solver for Robotic Applications
<https://doi.org/10.1109/LRA.2019.2926664>`_.
"""

from typing import Optional

import numpy as np
import qpSWIFT


def qpswift_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
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
            & A x = b
        \\end{array}\\end{split}

    using `qpSWIFT <https://github.com/qpSWIFT/qpSWIFT>`__.

    Note
    ----
    This solver does not handle problems without inequality constraints yet.

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
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    .. _qpSWIFT rank assumptions:

    **Rank assumptions:** qpSWIFT requires the QP matrices to satisfy the

    .. math::

        \\begin{split}\\begin{array}{cc}
        \\mathrm{rank}(A) = p
        &
        \\mathrm{rank}([P\\ A^T\\ G^T]) = n
        \\end{array}\\end{split}

    where :math:`p` is the number of rows of :math:`A` and :math:`n` is the
    number of optimization variables. This is the same requirement as
    :func:`cvxopt_solve_qp`, however qpSWIFT does not perform rank checks as it
    prioritizes performance. If the solver fails on your problem, try running
    CVXOPT on it for rank checks.

    Notes
    -----
    The option dictionary accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - MAXITER
         - maximum number of iterations needed
       * - ABSTOL
         - absolute tolerance
       * - RELTOL
         - relative tolerance
       * - SIGMA
         - maximum centering allowed

    If a verbose output shows that the maximum number of iterations is reached,
    check e.g. (1) the rank of your equality constraint matrix and (2) that
    your inequality constraint matrix does not have zero rows.

    As qpSWIFT does not sanity check its inputs, it should be used with a
    little more care than the other solvers. For instance, make sure you don't
    have zero rows in your input matrices, as it can `make the solver
    numerically unstable <https://github.com/qpSWIFT/qpSWIFT/issues/3>`_.

    Notes
    -----
    All other keyword arguments are forwarded as options to the qpSWIFT solver.
    For instance, you can call ``qpswift_solve_qp(P, q, G, h, ABSTOL=1e-5)``.
    See the solver documentation for details.
    """
    if initvals is not None:
        print("qpSWIFT: note that warm-start values ignored by wrapper")
    result: dict = {}
    kwargs.update(
        {
            "OUTPUT": 1,  # include "sol" and "basicInfo"
            "VERBOSE": 1 if verbose else 0,
        }
    )
    if G is not None and h is not None:
        if A is not None and b is not None:
            result = qpSWIFT.run(q, h, P, G, A, b, kwargs)
        else:  # no equality constraint
            result = qpSWIFT.run(q, h, P, G, opts=kwargs)
    else:  # no inequality constraint
        # See https://qpswift.github.io/index.html#updates
        raise NotImplementedError(
            "QP without inequality constraints is still WIP for qpSWIFT"
        )
    exit_flag = result["basicInfo"]["ExitFlag"]
    if exit_flag != 0:
        return None
    return result["sol"]
