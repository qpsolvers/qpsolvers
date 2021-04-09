#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stéphane Caron <stephane.caron@normalesup.org>
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

"""Quadratic programming solvers in Python with a unified API"""

from typing import Any, Dict, Optional

from numpy import concatenate, dot, eye, hstack, ones, ndarray, vstack, zeros

__version__ = "1.6.1"

available_solvers = []
dense_solvers = []
sparse_solvers = []
__solve_function__: Dict[str, Any] = {}


# CVXOPT
# ======

try:
    from .cvxopt_ import cvxopt_solve_qp

    __solve_function__["cvxopt"] = cvxopt_solve_qp
    available_solvers.append("cvxopt")
    dense_solvers.append("cvxopt")
except ImportError:
    pass

# CVXPY
# =====

try:
    from .cvxpy_ import cvxpy_solve_qp

    __solve_function__["cvxpy"] = cvxpy_solve_qp
    available_solvers.append("cvxpy")
    sparse_solvers.append("cvxpy")
except ImportError:
    pass

# ECOS
# ====

try:
    from .ecos_ import ecos_solve_qp

    __solve_function__["ecos"] = ecos_solve_qp
    available_solvers.append("ecos")
    dense_solvers.append("ecos")  # considered dense as it calls cholesky(P)
except ImportError:
    pass

# Gurobi
# ======

try:
    from .gurobi_ import gurobi_solve_qp

    __solve_function__["gurobi"] = gurobi_solve_qp
    available_solvers.append("gurobi")
    sparse_solvers.append("gurobi")
except ImportError:
    pass

# MOSEK
# =====

try:
    from .mosek_ import mosek_solve_qp

    __solve_function__["mosek"] = mosek_solve_qp
    available_solvers.append("mosek")
    sparse_solvers.append("mosek")
except ImportError:
    pass

# OSQP
# ====

try:
    from .osqp_ import osqp_solve_qp

    __solve_function__["osqp"] = osqp_solve_qp
    available_solvers.append("osqp")
    sparse_solvers.append("osqp")
except ImportError:
    pass

# qpOASES
# =======

try:
    from .qpoases_ import qpoases_solve_qp

    __solve_function__["qpoases"] = qpoases_solve_qp
    available_solvers.append("qpoases")
    dense_solvers.append("qpoases")
except ImportError:
    pass

# quadprog
# ========

try:
    from .quadprog_ import quadprog_solve_qp

    __solve_function__["quadprog"] = quadprog_solve_qp
    available_solvers.append("quadprog")
    dense_solvers.append("quadprog")
except ImportError:
    pass


class SolverNotFound(Exception):
    pass


def check_problem_constraints(G, h, A, b) -> None:
    """
    Check that problem constraint matrices and vectors are correctly defined.

    Parameters
    ----------
    G : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix
        Linear inequality matrix.
    h : numpy.array
        Linear inequality vector.
    A : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix
        Linear equality matrix.
    b : numpy.array
        Linear equality vector.

    Raises
    ------
    ValueError
        If the constraints are not properly defined.
    """
    if G is None and h is not None:
        raise ValueError("incomplete inequality constraint (missing h)")
    if G is not None and h is None:
        raise ValueError("incomplete inequality constraint (missing G)")
    if A is None and b is not None:
        raise ValueError("incomplete equality constraint (missing b)")
    if A is not None and b is None:
        raise ValueError("incomplete equality constraint (missing A)")


def solve_qp(
    P,
    q,
    G=None,
    h=None,
    A=None,
    b=None,
    lb=None,
    ub=None,
    solver="quadprog",
    initvals=None,
    sym_proj=False,
    verbose=False,
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
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using one of the available QP solvers.

    Parameters
    ----------
    P : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix
        Symmetric quadratic-cost matrix (most solvers require it to be definite
        as well).
    q : numpy.array
        Quadratic-cost vector.
    G : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear inequality matrix.
    h : numpy.array, optional
        Linear inequality vector.
    A : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear equality matrix.
    b : numpy.array, optional
        Linear equality vector.
    lb: numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Lower bound constraint vector.
    ub: numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Upper bound constraint vector.
    solver : string, optional
        Name of the QP solver, to choose in ``qpsolvers.available_solvers``.
    initvals : array, optional
        Vector of initial `x` values used to warm-start the solver.
    sym_proj : bool, optional
        Set to `True` when the `P` matrix provided is not symmetric.
    verbose : bool, optional
        Set to `True` to print out extra information.

    Returns
    -------
    x : array or None
        Optimal solution if found, None otherwise.

    Raises
    ------
    ValueError
        If the problem is not correctly defined.

    Note
    ----
    Extra keyword arguments given to this function are forwarded to the
    underlying solvers. For example, OSQP has a setting `eps_abs` which we can
    provide by ``solve_qp(P, q, G, h, solver='osqp', eps_abs=1e-4)``.

    Notes
    -----
    In quadratic programming, the matrix `P` should be symmetric. Many solvers
    (including CVXOPT, OSQP and quadprog) leverage this property and may return
    erroneous results when it is not the case. You can set ``sym_proj=True`` to
    project `P` on its symmetric part, at the cost of some computation time.
    """
    if sym_proj:
        P = 0.5 * (P + P.transpose())
    if isinstance(A, ndarray) and A.ndim == 1:
        A = A.reshape((1, A.shape[0]))
    if isinstance(G, ndarray) and G.ndim == 1:
        G = G.reshape((1, G.shape[0]))
    check_problem_constraints(G, h, A, b)
    if lb is not None:
        if G is None:
            G = -eye(len(q))
            h = -lb
        else:  # G is not None and h is not None
            G = concatenate((G, -eye(len(q))), 0)
            h = concatenate((h, -lb))
    if ub is not None:
        if G is None:
            G = eye(len(q))
            h = ub
        else:  # G is not None and h is not None
            G = concatenate((G, eye(len(q))), 0)
            h = concatenate((h, ub))
    args = P, q, G, h, A, b
    kwargs["initvals"] = initvals
    kwargs["verbose"] = verbose
    try:
        return __solve_function__[solver](*args, **kwargs)
    except KeyError:
        raise SolverNotFound(f"solver '{solver}' is not available")


def solve_safer_qp(
    P,
    q,
    G,
    h,
    sw: float,
    reg: float = 1e-8,
    solver="mosek",
    initvals=None,
    sym_proj=False,
) -> Optional[ndarray]:
    """
    Solve the Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\mbox{minimize} &
                \\frac{1}{2} x^T P x + q^T x +
                \\frac{1}{2} \\mathit{reg} \\|s\\|^2 - \\mathit{sw} \\1^T s
                \\\\
            \\mbox{subject to}
                & G x \\leq h
        \\end{array}\\end{split}

    Slack variables `s` are increased by an additional term in the cost
    function, so that the solution of this "safer" QP is further inside the
    constraint region.

    Parameters
    ----------
    P : numpy.array
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality matrix.
    h : numpy.array
        Linear inequality vector.
    sw : float
        Weight of the linear cost on slack variables. Higher values bring the
        solution further inside the constraint region but override the
        minimization of the original objective.
    reg : float, optional
        Regularization term :math:`(1/2) \\epsilon` in the cost function. Set
        this parameter as small as possible (e.g. 1e-8), and increase it in
        case of numerical instability.
    solver : string, optional
        Name of the QP solver to use (default is MOSEK).
    initvals : array, optional
        Vector of initial `x` values used to warm-start the solver.
    sym_proj : bool, optional
        Set to `True` when the `P` matrix provided is not symmetric.

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the relaxed QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.

    Notes
    -----
    This method can be found in the Inverse Kinematics resolution of Nozawa et
    al. (Humanoids 2016). It also appears in earlier works such as the
    "optimally safe" tension distribution algorithm of Borgstrom et al. (IEEE
    Transactions on Robotics, 2009).
    """
    assert solver in dense_solvers, "only available for dense solvers, for now"

    n, m = P.shape[0], G.shape[0]
    E, Z = eye(m), zeros((m, n))
    P2 = vstack([hstack([P, Z.T]), hstack([Z, reg * eye(m)])])
    q2 = hstack([q, -sw * ones(m)])
    G2 = hstack([Z, E])
    h2 = zeros(m)
    A2 = hstack([G, -E])
    b2 = h
    x = solve_qp(
        P2, q2, G2, h2, A2, b2, solver=solver, initvals=initvals, sym_proj=sym_proj
    )
    if x is None:
        return None
    return x[:n]


def solve_ls(
    R,
    s,
    G=None,
    h=None,
    A=None,
    b=None,
    lb=None,
    ub=None,
    W=None,
    solver="quadprog",
    initvals=None,
    sym_proj=False,
    verbose=False,
    **kwargs,
) -> Optional[ndarray]:
    """
    Solve a constrained weighted linear Least Squares problem defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\mbox{minimize} &
                \\frac12 \\| R x - s \\|^2_W
                = \\frac12 (R x - s)^T W (R x - s) \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using one of the available QP solvers.

    Parameters
    ----------
    R : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix
        Symmetric matrix of the cost function (most solvers require it to be
        definite).
    s : numpy.array
        Vector term of the cost function.
    G : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear inequality matrix.
    h : numpy.array, optional
        Linear inequality vector.
    A : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Linear equality matrix.
    b : numpy.array, optional
        Linear equality vector.
    lb: numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Lower bound constraint vector.
    ub: numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Upper bound constraint vector.
    W : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix, optional
        Definite symmetric weight matrix used to define the norm of the cost
        function. The standard L2 norm (W = Identity) is used by default.
    solver : string, optional
        Name of the QP solver, to choose in ``qpsolvers.available_solvers``.
    initvals : array, optional
        Vector of initial `x` values used to warm-start the solver.
    sym_proj : bool, optional
        Set to `True` when the `R` matrix provided is not symmetric.
    verbose : bool, optional
        Set to `True` to print out extra information.

    Returns
    -------
    x : array or None
        Optimal solution if found, None otherwise.

    Note
    ----
    Extra keyword arguments given to this function are forwarded to the
    underlying solvers. For example, OSQP has a setting `eps_abs` which we can
    provide by ``solve_ls(R, s, G, h, solver='osqp', eps_abs=1e-4)``.
    """
    if sym_proj:
        R = 0.5 * (R + R.transpose())
    WR = R if W is None else dot(W, R)
    P = dot(R.transpose(), WR)
    q = -dot(s.transpose(), WR)
    return solve_qp(
        P,
        q,
        G,
        h,
        A,
        b,
        lb,
        ub,
        solver=solver,
        initvals=initvals,
        sym_proj=False,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "__version__",
    "available_solvers",
    "cvxopt_solve_qp",
    "cvxpy_solve_qp",
    "dense_solvers",
    "gurobi_solve_qp",
    "mosek_solve_qp",
    "qpoases_solve_qp",
    "quadprog_solve_qp",
    "solve_ls",
    "solve_qp",
    "solve_safer_qp",
    "sparse_solvers",
]
