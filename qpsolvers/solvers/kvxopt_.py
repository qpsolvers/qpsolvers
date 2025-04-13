#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `KVXOPT <https://github.com/sanurielf/kvxopt>`__.

KVXOPT is a fork from CVXOPT including more SuiteSparse functions
and KLU sparse matrix solver. As CVXOPT, it is a free, open-source
interior-point solver.
"""

import warnings
from typing import Dict, Optional, Union

import kvxopt
import numpy as np
import scipy.sparse as spa
from kvxopt.solvers import qp

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError, SolverError
from ..problem import Problem
from ..solution import Solution

kvxopt.solvers.options["show_progress"] = False  # disable default verbosity


def __to_cvxopt(
    M: Union[np.ndarray, spa.csc_matrix],
) -> Union[kvxopt.matrix, kvxopt.spmatrix]:
    """Convert matrix to CVXOPT format.

    Parameters
    ----------
    M :
        Matrix in NumPy or CVXOPT format.

    Returns
    -------
    :
        Matrix in CVXOPT format.
    """
    if isinstance(M, np.ndarray):
        __infty__ = 1e10  # 1e20 tends to yield division-by-zero errors
        M_noinf = np.nan_to_num(M, posinf=__infty__, neginf=-__infty__)
        return kvxopt.matrix(M_noinf)
    coo = M.tocoo()
    return kvxopt.spmatrix(
        coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape
    )


def kvxopt_solve_problem(
    problem: Problem,
    solver: Optional[str] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    r"""Solve a quadratic program using KVXOPT.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    solver :
        Set to 'mosek' to run MOSEK rather than KVXOPT.
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
    ProblemError
        If the KVXOPT rank assumption is not satisfied.

    SolverError
        If KVXOPT failed with an error.

    Note
    ----
    .. _KVXOPT rank assumptions:

    **Rank assumptions:** KVXOPT requires the QP matrices to satisfy the

    .. math::

        \begin{split}\begin{array}{cc}
        \mathrm{rank}(A) = p
        &
        \mathrm{rank}([P\ A^T\ G^T]) = n
        \end{array}\end{split}

    where :math:`p` is the number of rows of :math:`A` and :math:`n` is the
    number of optimization variables. See the "Rank assumptions" paragraph in
    the report `The CVXOPT linear and quadratic cone program solvers
    <http://www.ee.ucla.edu/~vandenbe/publications/coneprog.pdf>`_ for details.

    Notes
    -----
    KVXOPT only considers the lower entries of :math:`P`, therefore it will use
    a different cost than the one intended if a non-symmetric matrix is
    provided.

    Keyword arguments are forwarded as options to KVXOPT. For instance, we can
    call ``kvxopt_solve_qp(P, q, G, h, u, abstol=1e-4, reltol=1e-4)``. KVXOPT
    options include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``abstol``
         - Absolute tolerance on the duality gap.
       * - ``feastol``
         - Tolerance on feasibility conditions, that is, on the primal
           residual.
       * - ``maxiters``
         - Maximum number of iterations.
       * - ``refinement``
         - Number of iterative refinement steps when solving KKT equations
       * - ``reltol``
         - Relative tolerance on the duality gap.

    Check out `Algorithm Parameters
    <https://cvxopt.org/userguide/coneprog.html#algorithm-parameters>`_ section
    of the solver documentation for details and default values of all solver
    parameters. See also [Caron2022]_ for a primer on the duality gap, primal
    and dual residuals.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(
            G, h, lb, ub, use_sparse=problem.has_sparse
        )

    args = [__to_cvxopt(P), __to_cvxopt(q)]
    constraints = {"G": None, "h": None, "A": None, "b": None}
    if G is not None and h is not None:
        constraints["G"] = __to_cvxopt(G)
        constraints["h"] = __to_cvxopt(h)
    if A is not None and b is not None:
        constraints["A"] = __to_cvxopt(A)
        constraints["b"] = __to_cvxopt(b)
    initvals_dict: Optional[Dict[str, kvxopt.matrix]] = None
    if initvals is not None:
        if "mosek" in kwargs:
            warnings.warn("MOSEK: warm-start values are ignored")
        initvals_dict = {"x": __to_cvxopt(initvals)}
    kwargs["show_progress"] = verbose

    try:
        res = qp(
            *args,
            solver=solver,
            initvals=initvals_dict,
            options=kwargs,
            **constraints,
        )
    except ValueError as exception:
        error = str(exception)
        if "Rank(A)" in error:
            raise ProblemError(error) from exception
        raise SolverError(error) from exception

    solution = Solution(problem)
    solution.extras = res
    solution.found = "optimal" in res["status"]
    solution.x = np.array(res["x"]).flatten()
    solution.y = (
        np.array(res["y"]).flatten() if b is not None else np.empty((0,))
    )
    if h is not None and res["z"] is not None:
        z_cvxopt = np.array(res["z"]).flatten()
        if z_cvxopt.size == h.size:
            z, z_box = split_dual_linear_box(z_cvxopt, lb, ub)
            solution.z = z
            solution.z_box = z_box
    else:  # h is None
        solution.z = np.empty((0,))
        solution.z_box = np.empty((0,))
    solution.obj = res["primal objective"]
    return solution


def kvxopt_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using KVXOPT.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h                \\
                & A x = b                   \\
                & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `KVXOPT <https://github.com/sanurielf/kvxopt>`__.

    Parameters
    ----------
    P :
        Symmetric cost matrix. Together with :math:`A` and :math:`G`, it should
        satisfy :math:`\mathrm{rank}([P\ A^T\ G^T]) = n`, see the rank
        assumptions below.
    q :
        Cost vector.
    G :
        Linear inequality matrix. Together with :math:`P` and :math:`A`, it
        should satisfy :math:`\mathrm{rank}([P\ A^T\ G^T]) = n`, see the
        rank assumptions below.
    h :
        Linear inequality vector.
    A :
        Linear equality constraint matrix. It needs to be full row rank, and
        together with :math:`P` and :math:`G` satisfy
        :math:`\mathrm{rank}([P\ A^T\ G^T]) = n`. See the rank assumptions
        below.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    solver :
        Set to 'mosek' to run MOSEK rather than KVXOPT.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ProblemError
        If the KVXOPT rank assumption is not satisfied.

    SolverError
        If KVXOPT failed with an error.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = kvxopt_solve_problem(
        problem, solver, initvals, verbose, **kwargs
    )
    return solution.x if solution.found else None
