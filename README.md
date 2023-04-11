# QP Solvers for Python

[![Build](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/test.yml?branch=master)](https://github.com/qpsolvers/qpsolvers/actions)
[![Coverage](https://coveralls.io/repos/github/qpsolvers/qpsolvers/badge.svg?branch=master)](https://coveralls.io/github/qpsolvers/qpsolvers?branch=master)
[![Documentation](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/docs.yml?branch=master&label=docs)](https://qpsolvers.github.io/qpsolvers/)
[![Downloads/month](https://pepy.tech/badge/qpsolvers/month)](https://pepy.tech/project/qpsolvers)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/qpsolvers.svg)](https://anaconda.org/conda-forge/qpsolvers)
[![PyPI version](https://img.shields.io/pypi/v/qpsolvers)](https://pypi.org/project/qpsolvers/)

Unified interface to convex Quadratic Programming (QP) solvers available in Python.

## Installation

### Using PyPI

```console
pip install qpsolvers
```

### Using <img src="https://s3.amazonaws.com/conda-dev/conda_logo.svg" height="18">

```console
conda install qpsolvers -c conda-forge
```

Check out the documentation for [Windows](https://qpsolvers.github.io/qpsolvers/installation.html#windows) instructions.

## Usage

The library provides a one-stop shop [`solve_qp`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp) function with a ``solver`` keyword argument to select the backend solver. It solves convex quadratic programs in standard form:

$$
\begin{split}
\begin{array}{ll}
\underset{x}{\mbox{minimize}}
    & \frac{1}{2} x^T P x + q^T x \\
\mbox{subject to}
    & G x \leq h \\
    & A x = b \\
    & lb \leq x \leq ub
\end{array}
\end{split}
$$

Vector inequalities apply coordinate by coordinate. The function returns the solution $x^\*$ found by the solver, or ``None`` in case of failure/unfeasible problem. All solvers require the problem to be convex, meaning the matrix $P$ should be [positive semi-definite](https://en.wikipedia.org/wiki/Definite_symmetric_matrix). Some solvers further require the problem to be strictly convex, meaning $P$ should be positive definite.

**Dual multipliers:** alternatively, the [`solve_problem`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_problem) function returns a more complete solution object containing both the primal solution and its corresponding dual multipliers.

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

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``. It is also possible to get dual multipliers at the solution, as shown in [this example](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#dual-multipliers).

## Solvers

| Solver | Keyword | Algorithm | API | License | Warm-start |
| ------ | ------- | --------- | --- | ------- |------------|
| [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) | ``clarabel`` | Interior point | Sparse | Apache-2.0 | ✖️ |
| [CVXOPT](http://cvxopt.org/) | ``cvxopt`` | Interior point | Dense | GPL-3.0 | ✔️ |
| [DAQP](https://github.com/darnstrom/daqp) | ``daqp`` | Active set | Dense | MIT | ✖️ |
| [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) | ``ecos`` | Interior point | Sparse | GPL-3.0 | ✖️ |
| [Gurobi](https://www.gurobi.com/) | ``gurobi`` | Interior point | Sparse | Commercial | ✖️ |
| [HiGHS](https://highs.dev/) | ``highs`` | Active set | Sparse | MIT | ✖️ |
| [MOSEK](https://mosek.com/) | ``mosek`` | Interior point | Sparse | Commercial | ✔️ |
| NPPro | ``nppro`` | Active set | Dense | Commercial | ✔️ |
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
  - Yes, there is also a [`solve_ls`](https://qpsolvers.github.io/qpsolvers/least-squares.html#qpsolvers.solve_ls) function.
- *I have a squared norm in my cost function, how can I apply a QP solver to my problem?*
  - You can [cast squared norms to QP matrices](https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html) and feed the result to [`solve_qp`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp).
- *I have a non-convex quadratic program. Is there a solver I can use?*
  - Unfortunately most available QP solvers are designed for convex problems (i.e. problems for which `P` is positive semidefinite). That's in a way expected, as solving non-convex QP problems is NP hard.
  - CPLEX has methods for solving non-convex quadratic problems to [either local or global optimality](https://www.ibm.com/docs/fr/icos/22.1.0?topic=qp-distinguishing-between-convex-nonconvex-qps). Notice that finding global solutions can be significantly slower than [finding local solutions](https://link.springer.com/chapter/10.1007/978-1-4613-0263-6_8).
  - Gurobi can find [global solutions to non-convex quadratic problems](https://www.gurobi.com/wp-content/uploads/2020-01-14_Non-Convex-Quadratic-Optimization-in-Gurobi-9.0-Webinar.pdf).
  - For a free non-convex solver, you can try the popular nonlinear solver [IPOPT](https://pypi.org/project/ipopt/) *e.g.* using [CasADi](https://web.casadi.org/).
  - A list of (convex/non-convex) quadratic programming software (not necessarily in Python) was compiled by [Nick Gould and Phillip Toint](https://www.numerical.rl.ac.uk/people/nimg/qp/qp.html).
- *I get the following [build error on Windows](https://github.com/qpsolvers/qpsolvers/issues/28) when running `pip install qpsolvers`.*
  - You will need to install the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to build all package dependencies.
- *Can I help?*
  - Absolutely! The first step is to install the library and use it. Report any bug in the [issue tracker](https://github.com/qpsolvers/qpsolvers/issues).
  - If you're a developer looking to hack on open source, check out the [contribution guidelines](https://github.com/qpsolvers/qpsolvers/blob/master/CONTRIBUTING.md) for suggestions.

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

# Contributing

We welcome contributions, see [Contributing](https://github.com/qpsolvers/qpsolvers/blob/master/CONTRIBUTING.md) for details.
