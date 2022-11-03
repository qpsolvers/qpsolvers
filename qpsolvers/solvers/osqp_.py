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
Solver interface for `OSQP <https://osqp.org/>`__.

The OSQP solver implements an Operator-Splitting method for large-scale convex
quadratic programming. It is designed for both dense and sparse problems, and
convexity is the only assumption it makes on problem data (for instance, it
does not make any rank assumption contrary to :ref:`CVXOPT <CVXOPT rank
assumptions>` or :ref:`qpSWIFT <qpSWIFT rank assumptions>`).
"""

from typing import Optional, Union
from warnings import warn

import osqp
import scipy.sparse as spa
from numpy import hstack, inf, ndarray, ones
from osqp import OSQP
from scipy.sparse import csc_matrix

from .conversions import linear_from_box_inequalities
from .typing import warn_about_sparse_conversion


def osqp_solve_qp(
    P: Union[ndarray, csc_matrix],
    q: ndarray,
    G: Optional[Union[ndarray, csc_matrix]] = None,
    h: Optional[ndarray] = None,
    A: Optional[Union[ndarray, csc_matrix]] = None,
    b: Optional[ndarray] = None,
    lb: Optional[ndarray] = None,
    ub: Optional[ndarray] = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[ndarray]:
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

    using `OSQP <https://github.com/oxfordcontrol/osqp>`_.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix.
    q :
        Quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
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

    Raises
    ------
    ValueError
        If the problem is clearly non-convex. See `this recommendation
        <https://osqp.org/docs/interfaces/status_values.html#status-values>`_.
        Note that OSQP may find the problem unfeasible if the problem is
        slightly non-convex (in this context, the meaning of "clearly" and
        "slightly" depends on how close the negative eigenvalues of :math:`P`
        are to zero).

    Note
    ----
    OSQP requires a symmetric `P` and won't check for errors otherwise. Check
    out this point if you `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.

    Notes
    -----
    Keyword arguments are forwarded to OSQP. For instance, we can call
    ``osqp_solve_qp(P, q, G, h, u, eps_abs=1e-8, eps_rel=0.0)``. OSQP settings
    include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``max_iter``
         - Maximum number of iterations.
       * - ``time_limit``
         - Run time limit in seconds, 0 to disable.
       * - ``eps_abs``
         - Absolute feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_rel``
         - Relative feasibility tolerance. See `Convergence
           <https://osqp.org/docs/solver/index.html#convergence>`__.
       * - ``eps_prim_inf``
         - Primal infeasibility tolerance.
       * - ``eps_dual_inf``
         - Dual infeasibility tolerance.
       * - ``polish``
         - Perform polishing. See `Polishing
           <https://osqp.org/docs/solver/#polishing>`_.

    Check out the `OSQP settings
    <https://osqp.org/docs/interfaces/solver_settings.html>`_ documentation for
    all available settings..

    Lower values for absolute or relative tolerances yield more precise
    solutions at the cost of computation time. See *e.g.* [tolprimer]_ for an
    overview of solver tolerances.
    """
    if isinstance(P, ndarray):
        warn_about_sparse_conversion("P")
        P = csc_matrix(P)
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    solver = OSQP()
    kwargs.update(
        {
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "polish": polish,
            "verbose": verbose,
        }
    )
    if A is not None and b is not None:
        if isinstance(A, ndarray):
            warn_about_sparse_conversion("A")
            A = csc_matrix(A)
        if G is not None and h is not None:
            l_inf = -inf * ones(len(h))
            qp_A = spa.vstack([G, A], format="csc")
            qp_l = hstack([l_inf, b])
            qp_u = hstack([h, b])
            solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, **kwargs)
        else:  # no inequality constraint
            solver.setup(P=P, q=q, A=A, l=b, u=b, **kwargs)
    elif G is not None and h is not None:
        if isinstance(G, ndarray):
            warn_about_sparse_conversion("G")
            G = csc_matrix(G)
        l_inf = -inf * ones(len(h))
        solver.setup(P=P, q=q, A=G, l=l_inf, u=h, **kwargs)
    else:  # no inequality nor equality constraint
        solver.setup(P=P, q=q, **kwargs)
    if initvals is not None:
        solver.warm_start(x=initvals)
    res = solver.solve()
    if hasattr(solver, "constant"):
        success_status = solver.constant("OSQP_SOLVED")
    else:  # more recent versions of OSQP
        success_status = osqp.constant("OSQP_SOLVED")
    if res.info.status_val != success_status:
        warn(f"OSQP exited with status '{res.info.status}'")
        return None
    return res.x
