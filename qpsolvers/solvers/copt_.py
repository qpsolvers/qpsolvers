#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors
# Copyright 2021 Dustin Kenefake

"""Solver interface for `COPT <https://www.shanshu.ai/solver>`__.

The COPT Optimizer suite ships several solvers for mathematical programming,
including problems that have linear constraints, bound constraints, integrality
constraints, cone constraints, or quadratic constraints.
It targets modern CPU/GPU architectures and multi-core processors,

See the :ref:`installation page <copt-install>` for additional instructions
on installing this solver.
"""

import warnings
from typing import Optional, Sequence, Union

import coptpy
import numpy as np
import scipy.sparse as spa
from coptpy import COPT

from ..problem import Problem
from ..solution import Solution


def copt_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using COPT.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        This argument is not used by COPT.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to COPT as parameters. For instance, we
    can call ``copt_solve_qp(P, q, G, h, u, FeasTol=1e-8,
    DualTol=1e-8)``. COPT settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasTol``
         - Primal feasibility tolerance.
       * - ``DualTol``
         - Dual feasibility tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://guide.coap.online/copt/en-doc/parameter.html>`_
    documentation for all available COPT parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    if initvals is not None:
        warnings.warn("warm-start values are ignored by this wrapper")

    env_config = coptpy.EnvrConfig()
    if not verbose:
        env_config.set("nobanner", "1")

    env = coptpy.Envr(env_config)
    model = env.createModel()

    if not verbose:
        model.setParam(COPT.Param.Logging, 0)
    for param, value in kwargs.items():
        model.setParam(param, value)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = np.eye(num_vars)
    x = model.addMVar(
        num_vars, lb=-COPT.INFINITY, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS
    )
    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None and h is not None:
        ineq_constr = model.addMConstr(
            G,  # type: ignore[arg-type]
            x,
            COPT.LESS_EQUAL,
            h,
        )
    if A is not None and b is not None:
        eq_constr = model.addMConstr(
            A,  # type: ignore[arg-type]
            x,
            COPT.EQUAL,
            b,
        )
    if lb is not None:
        lb_constr = model.addMConstr(identity, x, COPT.GREATER_EQUAL, lb)
    if ub is not None:
        ub_constr = model.addMConstr(identity, x, COPT.LESS_EQUAL, ub)
    objective = 0.5 * (x @ P @ x) + q @ x  # type: ignore[operator]
    model.setObjective(objective, sense=COPT.MINIMIZE)
    model.solve()

    solution = Solution(problem)
    solution.extras["status"] = model.status
    solution.found = model.status in (COPT.OPTIMAL, COPT.IMPRECISE)
    if solution.found:
        # COPT v8.0.0+ Changed the default Python matrix modeling API from
        # `numpy` to its own implementation. `coptpy.NdArray` does not support
        # operators such as ">=", so convert to `np.ndarray`
        solution.x = __to_numpy(x.X)  # type: ignore[attr-defined]
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)
    return solution


def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[coptpy.MConstr],
    eq_constr: Optional[coptpy.MConstr],
    lb_constr: Optional[coptpy.MConstr],
    ub_constr: Optional[coptpy.MConstr],
) -> None:
    solution.z = (
        __to_numpy(-ineq_constr.Pi)  # type: ignore[attr-defined]
        if ineq_constr is not None
        else np.empty((0,))
    )
    solution.y = (
        __to_numpy(-eq_constr.Pi)  # type: ignore[attr-defined]
        if eq_constr is not None
        else np.empty((0,))
    )
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = __to_numpy(
            -ub_constr.Pi - lb_constr.Pi  # type: ignore[attr-defined]
        )
    elif ub_constr is not None:  # lb_constr is None
        solution.z_box = __to_numpy(
            -ub_constr.Pi  # type: ignore[attr-defined]
        )
    elif lb_constr is not None:  # ub_constr is None
        solution.z_box = __to_numpy(
            -lb_constr.Pi  # type: ignore[attr-defined]
        )
    else:  # lb_constr is None and ub_constr is None
        solution.z_box = np.empty((0,))


def copt_solve_qp(
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
    r"""Solve a quadratic program using COPT.

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

    It is solved using `COPT <https://www.shanshu.ai/solver>`__.

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
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        This argument is not used by COPT.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to COPT as parameters. For instance, we
    can call ``copt_solve_qp(P, q, G, h, u, FeasTol=1e-8,
    DualTol=1e-8)``. COPT settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasTol``
         - Primal feasibility tolerance.
       * - ``DualTol``
         - Dual feasibility tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://guide.coap.online/copt/en-doc/parameter.html>`_
    documentation for all available COPT parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = copt_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None


def __to_numpy(
    array_like: Union[
        coptpy.NdArray, np.ndarray, float, int, Sequence[Union[float, int]]
    ],
) -> np.ndarray:
    """Convert COPT NdArray or array-like objects to numpy ndarray.

    This function ensures compatibility with COPT v8+, which changed the
    default Python matrix modeling API from numpy to its own implementation
    (``coptpy.NdArray``).

    Parameters
    ----------
    array_like :
        Input array to convert. Supported types:
        - ``coptpy.NdArray`` from COPT v8+ (converted via ``tonumpy()``)
        - ``np.ndarray`` (returned as-is to avoid redundant copy)
        - Scalar values (float, int) → converted to 1-element 1D numpy array
        - Sequence types (list, tuple) of floats/ints → converted to 1D numpy
          array

    Returns
    -------
    :
        Numpy array representation of the input (1D for scalars/sequences, same
        shape for COPT/numpy arrays).

    Raises
    ------
    TypeError
        If the input type is not supported (e.g., dict, None, non-numeric
        sequence).
    RuntimeError
        If conversion from coptpy.NdArray to numpy fails (e.g., invalid COPT
        array).

    Notes
    -----
    COPT v8.0.0+ uses ``coptpy.NdArray`` by default, which does not support
    operators such as ``>=``. This function converts such arrays to
    ``np.ndarray`` for further processing.
    Numpy arrays are returned as-is to avoid unnecessary memory copies.

    Examples
    --------
    >>> # Convert COPT NdArray to numpy (when coptpy is available)
    >>> # copt_array = coptpy.NdArray([1.0, 2.0, 3.0])  # doctest: +SKIP
    >>> # np_array = __to_numpy(copt_array)  # doctest: +SKIP
    >>> # isinstance(np_array, np.ndarray)  # doctest: +SKIP
    >>> # True  # doctest: +SKIP

    >>> # Convert scalar to 1D numpy array
    >>> __to_numpy(5.0).shape
    (1,)

    >>> # Convert list to numpy array
    >>> __to_numpy([1, 2, 3]).shape
    (3,)
    """
    if array_like is None:
        raise TypeError(
            "Input 'array_like' cannot be None. Supported types: "
            "coptpy.NdArray, np.ndarray, float, int, list/tuple of numbers."
        )

    if isinstance(array_like, np.ndarray):
        return array_like

    if isinstance(array_like, coptpy.NdArray):
        try:
            return array_like.tonumpy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert coptpy.NdArray to numpy array: {str(e)}"
            ) from e

    try:
        if isinstance(array_like, (int, float)):
            return np.asarray([array_like])
        if isinstance(array_like, (list, tuple)):
            return np.asarray(array_like)
    except Exception as e:
        raise RuntimeError(
            "Failed to convert input to numpy array. Input type: "
            f"{type(array_like).__name__}, error: {str(e)}"
        ) from e

    raise TypeError(
        f"Unsupported type '{type(array_like).__name__}' for 'array_like'. "
        f"Supported types: coptpy.NdArray, np.ndarray, float, int, list/tuple "
        "of numbers."
    )
