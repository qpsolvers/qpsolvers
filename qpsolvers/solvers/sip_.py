#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `SIP`_.

.. _SIP: https://github.com/joaospinto/sip_python

SIP is a general NLP solver based. It is based on the barrier augmented
Lagrangian method, which combines the interior point and augmented Lagrangian
methods. If you are using SIP in a scientific work, consider citing the
corresponding GitHub repository (or paper, if one has been released).
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
import sip_python as sip

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution

_cvxopt_available = True
try:
    from cvxopt import amd, spmatrix  # noqa: F401
except ImportError:
    _cvxopt_available = False


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
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Notes
    -----
    All other keyword arguments are forwarded as options to SIP. For
    instance, you can call ``sip_solve_qp(P, q, G, h, eps_abs=1e-6)``.
    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - max_iterations
         - The maximum number of iterations the solver can do.
       * - max_ls_iterations
         - The maximum cumulative number of line search iterations.
       * - min_iterations_for_convergence
         - The least number of iterations until we can declare convergence.
       * - num_iterative_refinement_steps
         - The number of iterative refinement steps.
       * - max_kkt_violation
         - The maximum allowed violation of the KKT system.
       * - max_merit_slope
         - The maximum allowed merit function slope.
       * - initial_regularization
         - The initial x-regularizatino to be applied on the LHS.
       * - regularization_decay_factor
         - The multiplicative decay of the x-regularization coefficient.
       * - tau
         - A parameter of the fraction-to-the-boundary rule.
       * - start_ls_with_alpha_s_max
         - Determines whether we start with alpha=alpha_s_max or alpha=1.
       * - initial_mu
         - The initial barrier function coefficient.
       * - mu_update_factor
         - Determines how much mu decreases per iteration.
       * - mu_min
         - The minimum barrier coefficient.
       * - initial_penalty_parameter
         - The initial penalty parameter of the Augmented Lagrangian.
       * - min_acceptable_constraint_violation_ratio
         - Least acceptable constraint violation ratio to not increase eta.
       * - penalty_parameter_increase_factor
         - By what factor to increase eta.
       * - penalty_parameter_decrease_factor
         - By what factor to decrease eta.
       * - max_penalty_parameter
         - The maximum allowed penalty parameter in the AL merit function.
       * - armijo_factor
         - Determines when we accept a line search step.
       * - line_search_factor
         - Determines how much to backtrack at each line search iteration.
       * - line_search_min_step_size
         - Determines when we declare a line search failure.
       * - min_merit_slope_to_skip_line_search
         - Min merit slope to skip the line search.
       * - dual_armijo_factor
         - Fraction of the primal merit decrease to allow on the dual update.
       * - min_allowed_merit_increase
         - The minimum allowed merit function increase in the dual update.
       * - enable_elastics
         - Whether to enable the usage of elastic variables.
       * - elastic_var_cost_coeff
         - Elastic variables cost coefficient.
       * - enable_line_search_failures
         - Halts the optimization process if a good step is not found.
       * - print_logs
         - Determines whether we should print the solver logs.
       * - print_line_search_logs
         - Determines whether we should print the line search logs.
       * - print_search_direction_logs
         - Whether we should print the search direction computation logs.
       * - print_derivative_check_logs
         - Whether to print derivative check logs when something looks off.
       * - only_check_search_direction_slope
         - Only derivative-check the search direction.
       * - assert_checks_pass
         - Handle checks with assert calls.

    This list may not be exhaustive.
    Check the `Settings` struct in the `solver code
    <https://github.com/joaospinto/sip/blob/main/sip/types.hpp>`__ for details.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(
            G, h, lb, ub, use_sparse=problem.has_sparse
        )
    n: int = q.shape[0]

    # SIP does not support A, b, G, and h to be None.
    G = G if G is not None else spa.csr_matrix(np.zeros((0, n)))
    A = A if A is not None else spa.csr_matrix(np.zeros((0, n)))
    h = np.zeros((0,)) if h is None else h
    b = np.zeros((0,)) if b is None else b

    # Remove any infs from h.
    G[np.isinf(h), :] = 0.0
    h[np.isinf(h)] = 1.0

    if not isinstance(P, spa.csr_matrix):
        P = spa.csc_matrix(P)
        if verbose:
            warnings.warn("Converted P to a csc_matrix.")
    if not isinstance(G, spa.csr_matrix):
        G = spa.csr_matrix(G)
        if verbose:
            warnings.warn("Converted G to a csr_matrix.")
    if not isinstance(A, spa.csr_matrix):
        A = spa.csr_matrix(A)
        if verbose:
            warnings.warn("Converted A to a csr_matrix.")

    P.eliminate_zeros()
    G.eliminate_zeros()
    A.eliminate_zeros()

    P_T = spa.csc_matrix(P.T)
    if (
        (P.indices != P_T.indices).any()
        or (P.indptr != P_T.indptr).any()
        or (P.data != P_T.data).any()
    ):
        raise ProblemError("P should be symmetric.")

    if G is None and h is not None:
        raise ProblemError(
            "Inconsistent inequalities: G is not set but h is set"
        )
    if G is not None and h is None:
        raise ProblemError("Inconsistent inequalities: G is set but h is None")
    if A is None and b is not None:
        raise ProblemError(
            "Inconsistent inequalities: A is not set but b is set"
        )
    if A is not None and b is None:
        raise ProblemError("Inconsistent inequalities: A is set but b is None")

    k = None
    if allow_non_psd_P:
        eigenvalues, _eigenvectors = spa.linalg.eigsh(P, k=1, which="SM")
        k = -min(eigenvalues[0], 0.0) + 1e-3
    else:
        k = 1e-6

    # hess_L = P + k * spa.eye(n);
    # the code below avoids potential index cancellations.
    hess_L = spa.coo_matrix(P)
    upp_hess_L_rows = np.concatenate([hess_L.row, np.arange(n)])
    upp_hess_L_cols = np.concatenate([hess_L.col, np.arange(n)])
    upp_hess_L_data = np.concatenate([hess_L.data, k * np.ones(n)])
    hess_L = spa.coo_matrix(
        (upp_hess_L_data, (upp_hess_L_rows, upp_hess_L_cols)), shape=P.shape
    )
    hess_L.sum_duplicates()
    upp_hess_L = spa.triu(hess_L.tocsc())

    qs = sip.QDLDLSettings()
    qs.permute_kkt_system = True
    qs.kkt_pinv = sip.get_kkt_perm_inv(
        P=hess_L,
        A=A,
        G=G,
    )

    pd = sip.ProblemDimensions()
    pd.x_dim = n
    pd.s_dim = h.shape[0]
    pd.y_dim = b.shape[0]
    pd.upper_hessian_lagrangian_nnz = upp_hess_L.nnz
    pd.jacobian_c_nnz = A.nnz
    pd.jacobian_g_nnz = G.nnz
    pd.kkt_nnz, pd.kkt_L_nnz = sip.get_kkt_and_L_nnzs(
        P=hess_L,
        A=A,
        G=G,
        perm_inv=qs.kkt_pinv,
    )
    pd.is_jacobian_c_transposed = True
    pd.is_jacobian_g_transposed = True

    vars = sip.Variables(pd)

    if initvals is not None:
        vars.x[:] = initvals
    else:
        vars.x[:] = 0.0

    vars.s[:] = 1.0
    vars.y[:] = 0.0
    vars.e[:] = 0.0
    vars.z[:] = 1.0

    ss = sip.Settings()
    ss.max_iterations = 100
    ss.max_ls_iterations = 1000
    ss.max_kkt_violation = 1e-8
    ss.max_merit_slope = 1e-16
    ss.penalty_parameter_increase_factor = 2.0
    ss.mu_update_factor = 0.5
    ss.mu_min = 1e-16
    ss.max_penalty_parameter = 1e16
    ss.assert_checks_pass = True

    ss.print_logs = verbose
    ss.print_line_search_logs = verbose
    ss.print_search_direction_logs = verbose
    ss.print_derivative_check_logs = False

    for key, value in kwargs.items():
        try:
            setattr(ss, key, value)
        except AttributeError:
            if verbose:
                warnings.warn(
                    f"Received an undefined solver setting {key}\
                    with value {value}"
                )

    def mc(mci: sip.ModelCallbackInput) -> sip.ModelCallbackOutput:
        mco = sip.ModelCallbackOutput()

        Px = P.T @ mci.x

        mco.f = 0.5 * np.dot(Px, mci.x) + np.dot(q, mci.x)
        mco.c = A @ mci.x - b
        mco.g = G @ mci.x - h

        mco.gradient_f = Px + q
        mco.jacobian_c = A
        mco.jacobian_g = G
        mco.upper_hessian_lagrangian = upp_hess_L

        return mco

    solver = sip.Solver(ss, qs, pd, mc)

    output = solver.solve(vars)

    solution = Solution(problem)
    solution.extras = {"sip_output": output, "sip_vars": vars}
    solution.found = output.exit_status == sip.Status.SOLVED
    solution.obj = 0.5 * np.dot(P.T @ vars.x, vars.x) + np.dot(q, vars.x)
    solution.x = vars.x
    solution.y = vars.y
    if h is not None and vars.z is not None:
        z_sip = np.array(vars.z)
        z, z_box = split_dual_linear_box(z_sip, lb, ub)
        solution.z = z
        solution.z_box = z_box
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

    It is solved using `SIP
    <https://github.com/joaospinto/sip>`__.

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
    verbose :
        Set to `True` to print out extra information.
    initvals :
        Warm-start guess vector. Not used.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = sip_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
