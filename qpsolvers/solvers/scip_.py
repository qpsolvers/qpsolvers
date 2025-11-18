"""Solver interface for `SCIP <https://github.com/scipopt/PySCIPOpt/>`__.

SCIP is currently one of the fastest academically developed solvers for mixed integer programming (MIP) and mixed integer nonlinear programming (MINLP).
In addition, SCIP provides a highly flexible framework for constraint integer programming and branch-cut-and-price.
It allows for total control of the solution process and the access of detailed information down to the guts of the solver.
"""

import warnings
from typing import Optional, Union

import pyscipopt
import numpy as np
import scipy.sparse as spa

from ..conversions import ensure_sparse_matrices
from ..problem import Problem
from ..solution import Solution


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
    Keyword arguments are forwarded to SCIP as options. For instance, we
    can call ``scip_solve_qp(P, q, G, h, u, primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8)``. SCIP settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``dual_feasibility_tolerance``
         - Dual feasibility tolerance.
       * - ``primal_feasibility_tolerance``
         - Primal feasibility tolerance.
       * - ``time_limit``
         - Run time limit in seconds.

    Check out the `SCIP documentation <https://scipopt.org/>`_
    for more information on the solver.
    """
    model = pyscipopt.Model()

    P, q, G, h, A, b, lb, ub = problem.unpack()

    P, G, A = ensure_sparse_matrices("scip", P, G, A)
    num_vars = P.shape[0]
    x = model.addMatrixVar(
        num_vars, ub=model.infinity(), vtype="C"
    )
    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None:
        ineq_constr = model.addMatrixCons(G@x <= h)
    if A is not None:
        eq_constr = model.addMatrixCons(A@x == b)
    if lb is not None:
        lb_constr = model.addMatrixCons(x >= lb)
    if ub is not None:
        ub_constr = model.addMatrixCons(x <= ub)

    if initvals:
        init_sol = model.createSol()
        for i, val in enumerate(initvals):
            model.setSolVal(init_sol, model.getVar(i), val)
        model.addSol(init_sol, "warm-start", True)

    if verbose:
        model.hideOutput(False)
    if not verbose:
        model.hideOutput()

    for option, value in kwargs.items():
        model.setOptionValue(option, value)
    
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, "minimize")
    model.optimize()

    if model.getNSols() > 0:
        solution = model.getBestSol()
    
    solution = Solution(problem)
    solution.extras["status"] = model.getStatus()
    solution.found = model.getNSols() > 0
    solution.x = np.array(model.getBestSol())

    if solution.found:
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)

    return solution

def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[pyscipopt.MatrixConstraint],
    eq_constr: Optional[pyscipopt.MatrixConstraint],
    lb_constr: Optional[pyscipopt.MatrixConstraint],
    ub_constr: Optional[pyscipopt.MatrixConstraint],
) -> None:
    solution.z = -ineq_constr.getDualsolVal() if ineq_constr is not None else np.empty((0,))
    solution.y = -eq_constr.getDualsolVal() if eq_constr is not None else np.empty((0,))
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = -ub_constr.getDualsolVal() - lb_constr.getDualsolVal()
    elif ub_constr is not None:  # lb_constr is None
        solution.z_box = -ub_constr.getDualsolVal()
    elif lb_constr is not None:  # ub_constr is None
        solution.z_box = -lb_constr.getDualsolVal()
    else:  # lb_constr is None and ub_constr is None
        solution.z_box = np.empty((0,))

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
    Keyword arguments are forwarded to SCIP as options. For instance, we
    can call ``scip_solve_qp(P, q, G, h, u, primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8)``. SCIP settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``dual_feasibility_tolerance``
         - Dual feasibility tolerance.
       * - ``primal_feasibility_tolerance``
         - Primal feasibility tolerance.
       * - ``time_limit``
         - Run time limit in seconds.

    Check out the `SCIP documentation <https://www.scipopt.org/doc-9.2.2/html/>`_
    for more information on the solver.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = scip_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
