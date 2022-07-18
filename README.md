# QP Solvers for Python

[**Installation**](https://github.com/stephane-caron/qpsolvers#installation)
| [**Usage**](https://github.com/stephane-caron/qpsolvers#usage)
| [**Example**](https://github.com/stephane-caron/qpsolvers#example)
| [**Solvers**](https://github.com/stephane-caron/qpsolvers#solvers)
| [**FAQ**](https://github.com/stephane-caron/qpsolvers#frequently-asked-questions)
| [**Benchmark**](https://github.com/stephane-caron/qpsolvers#benchmark)

[![Build](https://img.shields.io/github/workflow/status/stephane-caron/qpsolvers/CI)](https://github.com/stephane-caron/qpsolvers/actions)
[![Coverage](https://coveralls.io/repos/github/stephane-caron/qpsolvers/badge.svg?branch=master)](https://coveralls.io/github/stephane-caron/qpsolvers?branch=master)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://scaron.info/doc/qpsolvers/)
[![PyPI version](https://img.shields.io/pypi/v/qpsolvers)](https://pypi.org/project/qpsolvers/)
![Status](https://img.shields.io/pypi/status/qpsolvers)

Unified interface to Quadratic Programming (QP) solvers available in Python.

📢 **With v2.0, the ``solver`` keyword argument has become mandatory.** There is no implicit default solver any more.

## Installation

To install both the library and a starter set of QP solvers:

```console
pip install qpsolvers[starter_solvers]
```

To only install the library:

```console
pip install qpsolvers
```
Check out the documentation for [Python 2](https://scaron.info/doc/qpsolvers/installation.html#python-2) or [Windows](https://scaron.info/doc/qpsolvers/installation.html#windows) instructions.

## Usage

The library provides a one-stop shop [`solve_qp`](https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp) function with a ``solver`` keyword argument to select the backend solver. It solves convex quadratic programs in standard form:

$$
\begin{split}
\begin{array}{ll}
\mbox{minimize}
    & \frac{1}{2} x^T P x + q^T x \\
\mbox{subject to}
    & G x \leq h \\
    & A x = b \\
    & lb \leq x \leq ub
\end{array}
\end{split}
$$

Vector inequalities are taken coordinate by coordinate. For most solvers, the matrix $P$ should be [positive definite](https://en.wikipedia.org/wiki/Definite_symmetric_matrix).

## Example

To solve a quadratic program, build the matrices that define it and call the ``solve_qp`` function:

```python
from numpy import array, dot
from qpsolvers import solve_qp

M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # this is a positive definite matrix
q = dot(array([3., 2., 3.]), M)
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.])
A = array([1., 1., 1.])
b = array([1.])

x = solve_qp(P, q, G, h, A, b, solver="osqp")
print("QP solution: x = {}".format(x))
```

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``.

## Solvers

The list of supported solvers currently includes:

| Solver | Keyword | Type | License | Warm-start |
| ------ | ------- | ---- | ------- |------------|
| [CVXOPT](http://cvxopt.org/) | ``cvxopt`` | Dense | GPL-3.0 | ✔️ |
| [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) | ``ecos`` | Sparse | GPL-3.0 | ✖️ |
| [Gurobi](https://www.gurobi.com/) | ``gurobi`` | Sparse | Commercial | ✖️ |
| [MOSEK](https://mosek.com/) | ``mosek`` | Sparse | Commercial | ✔️ |
| [OSQP](https://github.com/oxfordcontrol/osqp) | ``osqp`` | Sparse | Apache-2.0 | ✔️ |
| [qpOASES](https://github.com/coin-or/qpOASES) | ``qpoases`` | Dense | LGPL-2.1 | ➖ |
| [qpSWIFT](https://github.com/qpSWIFT/qpSWIFT) | ``qpswift`` | Sparse | GPL-3.0 | ✖️ |
| [quadprog](https://pypi.python.org/pypi/quadprog/) | ``quadprog`` | Dense | GPL-2.0 | ✖️ |
| [SCS](https://github.com/cvxgrp/scs) | ``scs`` | Sparse | MIT | ✔️ |

## Frequently Asked Questions

- *Can I print the list of solvers available on my machine?*
  - Absolutely: ``print(qpsolvers.available_solvers)``
- *Is it possible to solve a least squares rather than a quadratic program?*
  - Yes, `qpsolvers` also provides a [`solve_ls`](https://scaron.info/doc/qpsolvers/least-squares.html#qpsolvers.solve_ls) function.
- *I have a squared norm in my cost function, how can I apply a QP solver to my problem?*
  - You can [cast squared norms to QP matrices](https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html) and feed the result to [`solve_qp`](https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp).
- *I have a non-convex quadratic program. Is there a solver I can use?*
  - Unfortunately most available QP solvers are designed for convex problems.
  - If your cost matrix *P* is semi-definite rather than definite, try OSQP.
  - If your problem has concave components, go for a nonlinear solver such as [IPOPT](https://pypi.org/project/ipopt/) *e.g.* using [CasADi](https://web.casadi.org/).
- *I get the following [build error on Windows](https://github.com/stephane-caron/qpsolvers/issues/28) when running `pip install qpsolvers`.*
  - You will need to install the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to build all package dependencies.

## Benchmark

On a [dense problem](examples/benchmark_dense_problem.py), the performance of all solvers (as measured by IPython's ``%timeit`` on an Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz) is:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| qpswift  | Dense  | 0.008     |
| quadprog | Dense  | 0.01      |
| qpoases  | Dense  | 0.02      |
| osqp     | Sparse | 0.03      |
| scs      | Sparse | 0.03      |
| ecos     | Sparse | 0.27      |
| cvxopt   | Dense  | 0.44      |
| gurobi   | Sparse | 1.74      |
| cvxpy    | Sparse | 5.71      |
| mosek    | Sparse | 7.17      |

On a [sparse problem](examples/benchmark_sparse_problem.py) with *n = 500* optimization variables, these performances become:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| osqp     | Sparse |    1      |
| qpswift  | Dense  |    2      |
| scs      | Sparse |    4      |
| cvxpy    | Sparse |   11      |
| mosek    | Sparse |   17      |
| ecos     | Sparse |   33      |
| cvxopt   | Dense  |   51      |
| gurobi   | Sparse |  221      |
| quadprog | Dense  |  427      |
| qpoases  | Dense  | 1560      |

On a [model predictive control](examples/model_predictive_control.py) problem for robot locomotion, we get:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| quadprog | Dense  | 0.03      |
| qpswift  | Dense  | 0.08      |
| qpoases  | Dense  | 0.36      |
| osqp     | Sparse | 0.48      |
| ecos     | Sparse | 0.69      |
| scs      | Sparse | 0.76      |
| cvxopt   | Dense  | 2.75      |
| cvxpy    | Sparse | 7.02      |

Finally, here is a small benchmark of [random dense problems](examples/benchmark_random_problems.py) (each data point corresponds to an average over 10 runs):

<img src="https://scaron.info/images/qp-benchmark-2022.png">

Note that performances of QP solvers largely depend on the problem solved. For instance, MOSEK performs an [automatic conversion to Second-Order Cone Programming (SOCP)](https://docs.mosek.com/8.1/pythonapi/prob-def-quadratic.html) which the documentation advises bypassing for better performance. Similarly, ECOS reformulates [from QP to SOCP](qpsolvers/solvers/convert_to_socp.py) and [works best on small problems](https://web.stanford.edu/%7Eboyd/papers/ecos.html).
