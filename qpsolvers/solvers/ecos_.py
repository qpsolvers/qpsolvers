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
Solver interface for `ECOS <https://github.com/embotech/ecos>`__.

ECOS is an interior-point solver for convex second-order cone programs (SOCPs).
designed specifically for embedded applications. ECOS is written in low
footprint, single-threaded, library-free ANSI-C and so runs on most embedded
platforms. For small problems, ECOS is faster than most existing SOCP solvers;
it is still competitive for medium-sized problems up to tens of thousands of
variables. If you are using ECOS in some academic work, consider citing the
corresponding paper [Domahidi2013]_.
"""

import warnings
from typing import Optional, Union

import numpy as np
from ecos import solve
from scipy import sparse as spa

from ..conversions import (
    linear_from_box_inequalities,
    socp_from_qp,
    split_dual_linear_box,
)
from ..problem import Problem
from ..solution import Solution

__exit_flag_meaning__ = {
    0: "OPTIMAL",
    1: "PINF: found certificate of primal infeasibility",
    2: "DING: found certificate of dual infeasibility",
    10: "INACC_OFFSET: inaccurate results",
    -1: "MAXIT: maximum number of iterations reached",
    -2: "NUMERICS: search direction is unreliable",
    -3: "OUTCONE: primal or dual variables got outside of cone",
    -4: "SIGINT: solver interrupted",
    -7: "FATAL: unknown solver problem",
}


def ecos_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b
        \\end{array}\\end{split}

    using `ECOS <https://github.com/embotech/ecos>`_.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    All other keyword arguments are forwarded as options to the ECOS solver.
    For instance, you can call ``qpswift_solve_qp(P, q, G, h, abstol=1e-5)``.

    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``feastol``
         -  Tolerance on the primal and dual residual.
       * - ``abstol``
         -  Absolute tolerance on the duality gap.
       * - ``reltol``
         -  Relative tolerance on the duality gap.
       * - ``feastol_inacc``
         -  Tolerance on the primal and dual residual if reduced precisions.
       * - ``abstol_inacc``
         - Absolute tolerance on the duality gap if reduced precision.
       * - ``reltolL_inacc``
         - Relative tolerance on the duality gap if reduced precision.
       * - ``max_iters``
         - Maximum numer of iterations.
       * - ``nitref``
         - Number of iterative refinement steps.

    See the `ECOS Python wrapper documentation
    <https://github.com/embotech/ecos-python#calling-ecos-from-python>`_ for
    more details. You can also check out [tolerances]_ for a primer on
    primal-dual residuals or the duality gap.
    """
    if initvals is not None:
        warnings.warn("warm-start values are ignored by this wrapper")
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub)
    kwargs.update(
        {
            "verbose": verbose,
        }
    )
    c_socp, G_socp, h_socp, dims = socp_from_qp(P, q, G, h)
    if A is not None:
        A_socp = spa.hstack([A, spa.csc_matrix((A.shape[0], 1))], format="csc")
        result = solve(c_socp, G_socp, h_socp, dims, A_socp, b, **kwargs)
    else:
        result = solve(c_socp, G_socp, h_socp, dims, **kwargs)
    flag = result["info"]["exitFlag"]
    if flag != 0:
        meaning = __exit_flag_meaning__.get(flag, "unknown exit flag")
        warnings.warn(f"ECOS returned exit flag {flag} ({meaning})")
        return Solution(problem)

    solution = Solution(problem)
    solution.x = result["x"][:-1]
    if A is not None:
        solution.y = result["y"]
    if G is not None:
        z_ecos = result["z"][: G.shape[0]]
        z, z_box = split_dual_linear_box(z_ecos, lb, ub)
        solution.z = z
        solution.z_box = z_box
    solution.extras = result["info"]
    return solution


def ecos_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
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
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b
        \\end{array}\\end{split}

    using `ECOS <https://github.com/embotech/ecos>`_.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    All other keyword arguments are forwarded as options to the ECOS solver.
    For instance, you can call ``qpswift_solve_qp(P, q, G, h, abstol=1e-5)``.

    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``feastol``
         -  Tolerance on the primal and dual residual.
       * - ``abstol``
         -  Absolute tolerance on the duality gap.
       * - ``reltol``
         -  Relative tolerance on the duality gap.
       * - ``feastol_inacc``
         -  Tolerance on the primal and dual residual if reduced precisions.
       * - ``abstol_inacc``
         - Absolute tolerance on the duality gap if reduced precision.
       * - ``reltolL_inacc``
         - Relative tolerance on the duality gap if reduced precision.
       * - ``max_iters``
         - Maximum numer of iterations.
       * - ``nitref``
         - Number of iterative refinement steps.

    See the `ECOS Python wrapper documentation
    <https://github.com/embotech/ecos-python#calling-ecos-from-python>`_ for
    more details. You can also check out [tolerances]_ for a primer on
    primal-dual residuals or the duality gap.
    """
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = ecos_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x
