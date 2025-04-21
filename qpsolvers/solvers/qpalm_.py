#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `QPALM`_.

.. _QPALM: https://github.com/kul-optec/QPALM

QPALM is a proximal augmented-Lagrangian solver for (possibly nonconvex)
quadratic programs, implemented in the C programming language. If you use QPALM
in a scientific work, consider citing the corresponding paper [Hermans2022]_.
"""

import warnings
from typing import Optional, Union

import numpy as np
import qpalm
import scipy.sparse as spa

from ..conversions import (
    combine_linear_box_inequalities,
    ensure_sparse_matrices,
)
from ..exceptions import ParamError
from ..problem import Problem
from ..solution import Solution


def qpalm_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using QPALM.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Raises
    ------
    ParamError
        If a warm-start value is given both in `initvals` and the `x` keyword
        argument.

    Note
    ----
    QPALM internally only uses the upper-triangular part of the cost matrix
    :math:`P`.

    Notes
    -----
    Keyword arguments are forwarded as "settings" to QPALM. For instance, we
    can call ``qpalm_solve_qp(P, q, G, h, u, eps_abs=1e-4, eps_rel=1e-4)``.

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``max_iter``
         - Maximum number of iterations.
       * - ``eps_abs``
         - Asbolute stopping criterion of the solver. See *e.g.* [Caron2022]_
           for an overview of solver tolerances.
       * - ``eps_rel``
         - Relative stopping criterion of the solver. See *e.g.* [Caron2022]_
           for an overview of solver tolerances.
       * - ``rho``
         - Tolerance scaling factor.
       * - ``theta``
         - Penalty update criterion parameter.
       * - ``delta``
         - Penalty update factor.
       * - ``sigma_max``
         - Penalty factor cap.
       * - ``proximal``
         - Boolean, use proximal method of multipliers or not.

    This list is not exhaustive. Check out the `solver documentation
    <https://kul-optec.github.io/QPALM/Doxygen/structqpalm_1_1Settings.html>`__
    for details.
    """
    if initvals is not None:
        if "x" in kwargs:
            raise ParamError(
                "Warm-start value specified in both `initvals` and `x` kwargs"
            )
        kwargs["x"] = initvals

    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices("qpalm", P, G, A)
    n: int = q.shape[0]

    Cx, ux, lx = combine_linear_box_inequalities(G, h, lb, ub, n, use_csc=True)
    if A is not None and b is not None:
        Cx = spa.vstack((A, Cx), format="csc") if Cx is not None else A
        lx = np.hstack((b, lx)) if lx is not None else b
        ux = np.hstack((b, ux)) if ux is not None else b
    m: int = Cx.shape[0] if Cx is not None else 0

    data = qpalm.Data(n, m)
    if Cx is not None:
        data.A = Cx
        data.bmax = ux
        data.bmin = lx
    data.Q = P
    data.q = q

    settings = qpalm.Settings()
    settings.verbose = verbose
    for key, value in kwargs.items():
        try:
            setattr(settings, key, value)
        except AttributeError:
            if verbose:
                warnings.warn(
                    f"Received an undefined solver setting {key}\
                    with value {value}"
                )

    solver = qpalm.Solver(data, settings)
    solver.solve()

    solution = Solution(problem)
    solution.extras = {"info": solver.info}
    solution.found = solver.info.status == "solved"
    solution.x = solver.solution.x
    m_eq: int = A.shape[0] if A is not None else 0
    m_leq: int = G.shape[0] if G is not None else 0
    solution.y = solver.solution.y[0:m_eq]
    solution.z = solver.solution.y[m_eq : m_eq + m_leq]
    solution.z_box = solver.solution.y[m_eq + m_leq :]
    return solution


def qpalm_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    lb: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    ub: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using QPALM.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `QPALM <https://github.com/kul-optec/QPALM>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
    q :
        Cost vector.
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
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpalm_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
