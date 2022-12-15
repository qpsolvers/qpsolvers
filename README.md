# QP Solvers for Python

[![Build](https://img.shields.io/github/workflow/status/stephane-caron/qpsolvers/CI)](https://github.com/stephane-caron/qpsolvers/actions)
[![Coverage](https://coveralls.io/repos/github/stephane-caron/qpsolvers/badge.svg?branch=master)](https://coveralls.io/github/stephane-caron/qpsolvers?branch=master)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?style=flat)](https://scaron.info/doc/qpsolvers/)
[![Downloads/month](https://pepy.tech/badge/qpsolvers/month)](https://pepy.tech/project/qpsolvers)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/qpsolvers.svg)](https://anaconda.org/conda-forge/qpsolvers)
[![PyPI version](https://img.shields.io/pypi/v/qpsolvers)](https://pypi.org/project/qpsolvers/)

Unified interface to Quadratic Programming (QP) solvers available in Python.

## Installation

### Using PyPI

To install both the library and a starter set of open-source QP solvers:

```console
pip install qpsolvers[open_source_solvers]
```

To install just the library:

```console
pip install qpsolvers
```

### Using <img src="https://s3.amazonaws.com/conda-dev/conda_logo.svg" height="18">

```console
conda install qpsolvers -c conda-forge
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

📢 **New with v2.7:** get dual multipliers at the solution using the [`solve_problem`](https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_problem) function.

## Example

To solve a quadratic program, build the matrices that define it and call the ``solve_qp`` function:

```python
import numpy as np
from qpsolvers import solve_qp

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = M.T @ M  # this is a positive definite matrix
q = np.array([3.0, 2.0, 3.0]) @ M
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0])
A = np.array([1.0, 1.0, 1.0])
b = np.array([1.0])

x = solve_qp(P, q, G, h, A, b, solver="proxqp")
print(f"QP solution: x = {x}")
```

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``. It is also possible to [get dual multipliers](https://scaron.info/doc/qpsolvers/quadratic-programming.html#dual-multipliers) at the solution.

## Solvers

| Solver | Keyword | Algorithm | API | License | Warm-start |
| ------ | ------- | --------- | --- | ------- |------------|
| [CVXOPT](http://cvxopt.org/) | ``cvxopt`` | Interior point | Dense | GPL-3.0 | ✔️ |
| [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) | ``ecos`` | Interior point | Sparse | GPL-3.0 | ✖️ |
| [Gurobi](https://www.gurobi.com/) | ``gurobi`` | Interior point | Sparse | Commercial | ✖️ |
| [HiGHS](https://highs.dev/) | ``highs`` | Active set | Sparse | MIT | ✖️ |
| [MOSEK](https://mosek.com/) | ``mosek`` | Interior point | Sparse | Commercial | ✔️ |
| [OSQP](https://osqp.org/) | ``osqp`` | Augmented Lagrangian | Sparse | Apache-2.0 | ✔️ |
| [ProxQP](https://github.com/Simple-Robotics/proxsuite) | ``proxqp`` | Augmented Lagrangian | Dense & Sparse | BSD-2-Clause | ✔️ |
| [qpOASES](https://github.com/coin-or/qpOASES) | ``qpoases`` | Active set | Dense | LGPL-2.1 | ➖ |
| [qpSWIFT](https://qpswift.github.io/) | ``qpswift`` | Interior point | Sparse | GPL-3.0 | ✖️ |
| [quadprog](https://pypi.python.org/pypi/quadprog/) | ``quadprog`` | Active set | Dense | GPL-2.0 | ✖️ |
| [SCS](https://www.cvxgrp.org/scs/) | ``scs`` | Augmented Lagrangian | Sparse | MIT | ✔️ |

Matrix arguments are NumPy arrays for dense solvers and SciPy Compressed Sparse Column (CSC) matrices for sparse ones.

## Frequently Asked Questions

- *Can I print the list of solvers available on my machine?*
  - Absolutely: ``print(qpsolvers.available_solvers)``
- *Is it possible to solve a least squares rather than a quadratic program?*
  - Yes, there is also a [`solve_ls`](https://scaron.info/doc/qpsolvers/least-squares.html#qpsolvers.solve_ls) function.
- *I have a squared norm in my cost function, how can I apply a QP solver to my problem?*
  - You can [cast squared norms to QP matrices](https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html) and feed the result to [`solve_qp`](https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp).
- *I have a non-convex quadratic program. Is there a solver I can use?*
  - Unfortunately most available QP solvers are designed for convex problems.
  - If your cost matrix *P* is semi-definite rather than definite, try OSQP.
  - If your problem has concave components, go for a nonlinear solver such as [IPOPT](https://pypi.org/project/ipopt/) *e.g.* using [CasADi](https://web.casadi.org/).
- *I get the following [build error on Windows](https://github.com/stephane-caron/qpsolvers/issues/28) when running `pip install qpsolvers`.*
  - You will need to install the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to build all package dependencies.
- *Can I help?*
  - Absolutely! The first step is to install the library and use it. Report any bug in the [issue tracker](https://github.com/stephane-caron/qpsolvers/issues).
  - If you're a developer looking to hack on open source, check out the [contribution guidelines](CONTRIBUTING.md) for suggestions.

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
| mosek    | Sparse | 7.17      |

On a [sparse problem](examples/benchmark_sparse_problem.py) with *n = 500* optimization variables, these performances become:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| osqp     | Sparse |    1      |
| qpswift  | Dense  |    2      |
| scs      | Sparse |    4      |
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

Finally, here is a small benchmark of [random dense problems](examples/benchmark_random_problems.py) (each data point corresponds to an average over 10 runs):

<img src="https://scaron.info/images/qp-benchmark-2022.png">

Note that performances of QP solvers largely depend on the problem solved. For instance, MOSEK performs an [automatic conversion to Second-Order Cone Programming (SOCP)](https://docs.mosek.com/8.1/pythonapi/prob-def-quadratic.html) which the documentation advises bypassing for better performance. Similarly, ECOS reformulates [from QP to SOCP](qpsolvers/solvers/conversions/socp_from_qp.py) and [works best on small problems](https://web.stanford.edu/%7Eboyd/papers/ecos.html).
