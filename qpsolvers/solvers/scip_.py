"""Solver interface for `SCIP <https://github.com/scipopt/PySCIPOpt/>`__.

SCIP is currently one of the fastest academically developed solvers for
mixed integer programming (MIP) and mixed integer nonlinear programming
(MINLP). In addition, SCIP provides a highly flexible framework for
constraint integer programming and branch-cut-and-price. It allows for
total control of the solution process and the access of detailed
information down to the guts of the solver.
"""

from typing import Optional, Union

import numpy as np
import pyscipopt
import scipy.sparse as spa
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from ..problem import Problem
from ..solution import Solution


def _to_dense(matrix):
    """Return a dense ndarray view of a matrix.

    pyscipopt's ``MatrixVariable`` does not support scipy.sparse matmul,
    so we densify problem matrices before building SCIP expressions.
    """
    if matrix is None:
        return None
    if spa.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def scip_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using PySCIPOpt.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to SCIP as parameters. For instance,
    we can call ``scip_solve_qp(P, q, G, h, limits_time=10)`` to set a
    time limit of 10 seconds. See the `SCIP parameter reference
    <https://www.scipopt.org/doc/html/PARAMETERS.php>`_ for the list of
    supported parameters.
    """
    model = pyscipopt.Model()
    model.hideOutput(not verbose)
    # Tighten SCIP's default tolerances for QP-grade accuracy.
    # Going below 1e-9 requires SCIP to be built with GMP.
    defaults = {
        "numerics/feastol": 1e-9,
        "numerics/dualfeastol": 1e-9,
        "limits/gap": 1e-9,
        "limits/absgap": 1e-9,
    }
    for option, value in defaults.items():
        if option not in kwargs:
            model.setParam(option, value)
    for option, value in kwargs.items():
        model.setParam(option, value)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    P = _to_dense(P)
    G = _to_dense(G)
    A = _to_dense(A)
    num_vars = P.shape[0]
    x = model.addMatrixVar(
        num_vars,
        lb=-model.infinity(),
        ub=model.infinity(),
        vtype="C",
    )

    ineq_cons = model.addMatrixCons(G @ x <= h) if G is not None else None
    eq_cons = model.addMatrixCons(A @ x == b) if A is not None else None
    lb_cons = model.addMatrixCons(x >= lb) if lb is not None else None
    ub_cons = model.addMatrixCons(x <= ub) if ub is not None else None

    if initvals is not None:
        init_sol = model.createSol()
        for i, val in enumerate(initvals):
            model.setSolVal(init_sol, x[i], float(val))
        model.addSol(init_sol, free=True)

    objective = 0.5 * (x @ P @ x) + q @ x
    set_nonlinear_objective(model, objective, "minimize")
    model.optimize()

    solution = Solution(problem)
    solution.extras["status"] = model.getStatus()
    solution.found = model.getNSols() > 0
    if solution.found:
        solution.x = np.asarray(model.getVal(x))
        _retrieve_dual(model, solution, ineq_cons, eq_cons, lb_cons, ub_cons)
    return solution


def _dual_values(
    model: pyscipopt.Model,
    cons: Optional["pyscipopt.MatrixConstraint"],
) -> np.ndarray:
    """Return an array of dual values for a matrix of linear constraints."""
    if cons is None:
        return np.empty((0,))
    values = np.zeros(cons.shape)
    flat = values.reshape(-1)
    for i, c in enumerate(cons.reshape(-1)):
        try:
            flat[i] = model.getDualsolLinear(c)
        except Exception:  # pragma: no cover - SCIP raised an error
            flat[i] = 0.0
    return values


def _retrieve_dual(
    model: pyscipopt.Model,
    solution: Solution,
    ineq_cons: Optional["pyscipopt.MatrixConstraint"],
    eq_cons: Optional["pyscipopt.MatrixConstraint"],
    lb_cons: Optional["pyscipopt.MatrixConstraint"],
    ub_cons: Optional["pyscipopt.MatrixConstraint"],
) -> None:
    solution.z = -_dual_values(model, ineq_cons)
    solution.y = -_dual_values(model, eq_cons)
    if lb_cons is None and ub_cons is None:
        solution.z_box = np.empty((0,))
    else:
        assert solution.x is not None
        num_vars = solution.x.shape[0]
        lb_vals = (
            _dual_values(model, lb_cons)
            if lb_cons is not None
            else np.zeros(num_vars)
        )
        ub_vals = (
            _dual_values(model, ub_cons)
            if ub_cons is not None
            else np.zeros(num_vars)
        )
        solution.z_box = -(ub_vals - lb_vals)


def scip_solve_qp(
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
    r"""Solve a quadratic program using SCIP.

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

    It is solved using `SCIP <https://scipopt.org/>`__.

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
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to SCIP as parameters. See the
    `SCIP parameter reference
    <https://www.scipopt.org/doc/html/PARAMETERS.php>`_ for the list of
    supported parameters.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = scip_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
