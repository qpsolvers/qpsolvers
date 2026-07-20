#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Solver interface for `SIP`_.

.. _SIP: https://github.com/joaospinto/sip_qp

SIP is a general NLP solver based. It is based on the barrier augmented
Lagrangian method, which combines the interior point and augmented Lagrangian
methods. If you are using SIP in a scientific work, consider citing the
corresponding GitHub repository (or paper, if one has been released).

**Warm-start:** this solver interface supports warm starting 🔥
"""

import time
import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
import sip_qp_python as sip
from scipy.sparse.csgraph import reverse_cuthill_mckee

try:
    from cvxopt import amd, spmatrix
except ImportError:
    amd = None
    spmatrix = None

from ..conversions import (
    put_infinite_inequalities_back,
    remove_infinite_inequalities,
)
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def _set_setting(settings, key, value, verbose: bool) -> bool:
    """Set one QP or underlying SIP setting."""
    if hasattr(settings, key):
        owner = settings
    elif hasattr(settings.sip, key):
        owner = settings.sip
    else:
        return False

    target = getattr(owner, key)
    if not isinstance(value, dict):
        setattr(owner, key, value)
        return True

    for nested_key, nested_value in value.items():
        if hasattr(target, nested_key):
            setattr(target, nested_key, nested_value)
        elif verbose:
            warnings.warn(
                f"Received an undefined SIP solver setting "
                f"{key}.{nested_key} with value {nested_value}"
            )
    return True


def _as_csc(matrix) -> spa.csc_matrix:
    result = spa.csc_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _as_csr(matrix, rows: int, columns: int) -> spa.csr_matrix:
    result = (
        spa.csr_matrix((rows, columns), dtype=np.float64)
        if matrix is None
        else spa.csr_matrix(matrix, dtype=np.float64)
    )
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _kkt_inverse_permutation(
    P: spa.csc_matrix,
    A: spa.csr_matrix,
    G: spa.csr_matrix,
    verbose: bool,
) -> np.ndarray:
    """Compute an AMD ordering, falling back to RCM."""
    n = P.shape[0]
    num_equalities = A.shape[0]
    num_inequalities = G.shape[0]
    hessian_pattern = P.copy()
    hessian_pattern.data.fill(1.0)
    hessian_pattern = hessian_pattern.maximum(hessian_pattern.T)
    hessian_pattern.setdiag(1.0)
    equality_inequality_zeros = spa.csr_matrix(
        (num_equalities, num_inequalities)
    )
    kkt_pattern = spa.vstack(
        (
            spa.hstack((hessian_pattern, A.T, G.T), format="csr"),
            spa.hstack(
                (
                    A,
                    spa.eye(num_equalities, format="csr"),
                    equality_inequality_zeros,
                ),
                format="csr",
            ),
            spa.hstack(
                (
                    G,
                    equality_inequality_zeros.T,
                    spa.eye(num_inequalities, format="csr"),
                ),
                format="csr",
            ),
        ),
        format="csr",
    )
    kkt_pattern.data.fill(1.0)
    if amd is not None and spmatrix is not None:
        kkt_coo = kkt_pattern.tocoo()
        cvxopt_pattern = spmatrix(
            kkt_coo.data,
            kkt_coo.row,
            kkt_coo.col,
            kkt_coo.shape,
        )
        permutation = np.asarray(
            amd.order(cvxopt_pattern), dtype=np.int32
        ).ravel()
    else:
        if verbose:
            warnings.warn(
                "cvxopt is not installed; using reverse Cuthill-McKee "
                "instead of approximate minimum degree."
            )
        permutation = reverse_cuthill_mckee(kkt_pattern, symmetric_mode=True)
    kkt_dimension = n + num_equalities + num_inequalities
    inverse_permutation = np.empty(kkt_dimension, np.int32)
    inverse_permutation[permutation] = np.arange(
        permutation.size, dtype=np.int32
    )
    return inverse_permutation


def sip_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    allow_non_psd_P: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using SIP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess for the primal solution.
    verbose :
        Set to ``True`` to print SIP logs.
    allow_non_psd_P :
        This argument is not used by SIP.

    Returns
    -------
    :
        Solution to the QP returned by SIP.

    Notes
    -----
    Additional keyword arguments configure ``sip_qp_python.Settings``.
    Structured setting groups are passed as dictionaries. For example:

    .. code-block:: python

        solve_problem(
            problem,
            solver="sip",
            max_iterations=500,
            termination={"max_absolute_duality_gap": 1e-8},
            scaling={"max_iterations": 20},
        )

    The following settings are supported:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``mode``
         - Select regularized, primal-proximal, or primal-dual-proximal IPM.
       * - ``max_iterations``
         - Maximum number of SIP iterations.
       * - ``num_iterative_refinement_steps``
         - Number of Newton-KKT iterative-refinement steps.
       * - ``assert_checks_pass``
         - Handle internal consistency-check failures with assertions.
       * - ``barrier``
         - Dictionary containing ``initial_mu``, ``mu_update_factor``,
           ``mu_min``, and ``mu_update_kappa``.
       * - ``penalty``
         - Dictionary containing penalty initialization, warm-start, update,
           and maximum-value settings.
       * - ``termination``
         - Dictionary containing absolute and relative QP residual and
           duality-gap tolerances.
       * - ``regularization``
         - Dictionary containing the initial, first-positive, and maximum
           regularization values, factorization-attempt limit, and increase
           and decrease factors.
       * - ``line_search``
         - Dictionary containing fraction-to-boundary, Armijo, backtracking,
           filter, line-search failure, and line-search limit settings.
       * - ``logging``
         - Dictionary controlling solver, line-search, search-direction, and
           derivative-check logs.
       * - ``scaling``
         - Dictionary containing ``max_iterations``, ``min_norm``,
           ``max_norm``, and ``convergence_tolerance`` for QP equilibration.
       * - ``eps_abs``
         - Set the absolute QP residual and duality-gap tolerances.
       * - ``eps_rel``
         - Set the relative QP primal, dual, and duality-gap tolerances.
       * - ``time_limit``
         - Maximum solve time in seconds.

    Variable bounds are passed to SIP natively rather than expanded into
    general inequalities. Exact fixed bounds are represented as equalities
    because SIP requires strict lower and upper bound intervals.
    """
    build_start_time = time.perf_counter()
    P, q, G_input, h_input, A_input, b_input, lb, ub = problem.unpack()
    n = q.shape[0]

    if (G_input is None) != (h_input is None):
        raise ProblemError("G and h must either both be set or both be None")
    if (A_input is None) != (b_input is None):
        raise ProblemError("A and b must either both be set or both be None")

    P = _as_csc(P)
    G = _as_csr(G_input, 0, n)
    A = _as_csr(A_input, 0, n)
    q = np.ascontiguousarray(q, dtype=np.float64)
    h = (
        np.zeros(0, dtype=np.float64)
        if h_input is None
        else np.ascontiguousarray(h_input, dtype=np.float64)
    )
    b = (
        np.zeros(0, dtype=np.float64)
        if b_input is None
        else np.ascontiguousarray(b_input, dtype=np.float64)
    )

    G, h, finite_h = remove_infinite_inequalities(G, h)
    h = np.ascontiguousarray(h)

    lower_bounds = (
        np.full(n, -np.inf, dtype=np.float64)
        if lb is None
        else np.array(lb, dtype=np.float64, order="C", copy=True)
    )
    upper_bounds = (
        np.full(n, np.inf, dtype=np.float64)
        if ub is None
        else np.array(ub, dtype=np.float64, order="C", copy=True)
    )
    fixed_bounds = (
        np.isfinite(lower_bounds)
        & np.isfinite(upper_bounds)
        & (lower_bounds == upper_bounds)
    )
    num_user_equalities = A.shape[0]
    if np.any(fixed_bounds):
        fixed_indices = np.flatnonzero(fixed_bounds)
        fixed_rows = spa.csr_matrix(
            (
                np.ones(fixed_indices.size),
                (np.arange(fixed_indices.size), fixed_indices),
            ),
            shape=(fixed_indices.size, n),
        )
        A = spa.vstack((A, fixed_rows), format="csr")
        b = np.concatenate((b, lower_bounds[fixed_indices]))
        lower_bounds[fixed_indices] = -np.inf
        upper_bounds[fixed_indices] = np.inf
    initial_primal = (
        np.zeros(n, dtype=np.float64)
        if initvals is None
        else np.ascontiguousarray(initvals, dtype=np.float64)
    )

    settings = sip.Settings()
    settings.sip.logging.print_logs = verbose
    settings.sip.logging.print_line_search_logs = verbose
    settings.sip.logging.print_search_direction_logs = verbose
    settings.sip.logging.print_derivative_check_logs = False

    eps_abs = kwargs.pop("eps_abs", None)
    if eps_abs is not None:
        settings.termination.max_absolute_residual = eps_abs
        settings.termination.max_absolute_duality_gap = eps_abs

    eps_rel = kwargs.pop("eps_rel", None)
    if eps_rel is not None:
        settings.termination.max_relative_residual = eps_rel
        settings.termination.max_relative_duality_gap = eps_rel

    time_limit = kwargs.pop("time_limit", float("inf"))
    inverse_permutation = _kkt_inverse_permutation(P, A, G, verbose)

    for key, value in kwargs.items():
        if not _set_setting(settings, key, value, verbose) and verbose:
            warnings.warn(
                f"Received an undefined solver setting {key} "
                f"with value {value}"
            )

    _ = allow_non_psd_P
    try:
        solver = sip.Solver(
            P,
            q,
            G,
            h,
            A,
            b,
            lower_bounds,
            upper_bounds,
            inverse_permutation,
            settings,
            time_limit,
        )
    except ValueError as error:
        raise ProblemError(str(error)) from error

    solve_start_time = time.perf_counter()
    result = solver.solve(initial_primal)
    solve_end_time = time.perf_counter()

    solution = Solution(problem)
    solution.extras = {"sip_result": result, "sip_output": result.info}
    solution.found = result.info.exit_status == sip.Status.SOLVED
    solution.x = np.array(result.x)
    result_y = np.array(result.y)
    solution.y = result_y[:num_user_equalities]
    solution.obj = 0.5 * solution.x.dot(P @ solution.x) + q.dot(solution.x)

    solution.z = put_infinite_inequalities_back(np.array(result.z), finite_h)
    if lb is not None or ub is not None:
        solution.z_box = np.array(result.z_box)
        solution.z_box[fixed_bounds] = result_y[num_user_equalities:]
    else:
        solution.z_box = np.empty(0)
    solution.build_time = solve_start_time - build_start_time
    solution.solve_time = solve_end_time - solve_start_time
    return solution


def sip_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    allow_non_psd_P: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using SIP.

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

    It is solved using `SIP <https://github.com/joaospinto/sip_qp>`__.

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
        Warm-start guess for the primal solution.
    allow_non_psd_P :
        This argument is not used by SIP.
    verbose :
        Set to ``True`` to print SIP logs.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Additional keyword arguments are forwarded to :func:`sip_solve_problem`.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = sip_solve_problem(
        problem,
        initvals,
        verbose,
        allow_non_psd_P,
        **kwargs,
    )
    return solution.x if solution.found else None
