# Quadratic Programming Solvers in Python

[![Build](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/test.yml?branch=master)](https://github.com/qpsolvers/qpsolvers/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/docs.yml?branch=master&label=docs)](https://qpsolvers.github.io/qpsolvers/)
[![Coverage](https://coveralls.io/repos/github/qpsolvers/qpsolvers/badge.svg?branch=master)](https://coveralls.io/github/qpsolvers/qpsolvers?branch=master)

This library provides a one-stop shop [`solve_qp`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp) function to solve convex quadratic programs:

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

Vector inequalities apply coordinate by coordinate. The function returns the primal solution $x^\*$ found by the backend QP solver, or ``None`` in case of failure/unfeasible problem. All solvers require the problem to be convex, meaning the matrix $P$ should be [positive semi-definite](https://en.wikipedia.org/wiki/Definite_symmetric_matrix). Some solvers further require the problem to be strictly convex, meaning $P$ should be positive definite.

**Dual multipliers:** there is also a [`solve_problem`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_problem) function that returns not only the primal solution, but also its dual multipliers and all other relevant quantities computed by the backend solver.

## Example

To solve a quadratic program, build the matrices that define it and call ``solve_qp``, selecting the backend QP solver via the ``solver`` keyword argument:

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

## Installation

### PyPI

[![PyPI version](https://img.shields.io/pypi/v/qpsolvers)](https://pypi.org/project/qpsolvers/)
[![PyPI downloads](https://pepy.tech/badge/qpsolvers/month)](https://pepy.tech/project/qpsolvers)

To install the library with open source QP solvers:

```console
pip install qpsolvers[open_source_solvers]
```

To install only the library itself:

```console
pip install qpsolvers
```

When imported, qpsolvers loads all the solvers it can find and lists them in ``qpsolvers.available_solvers``.

### Conda

[![Conda version](https://anaconda.org/conda-forge/qpsolvers/badges/version.svg)](https://anaconda.org/conda-forge/qpsolvers)
[![Conda downloads](https://anaconda.org/conda-forge/qpsolvers/badges/downloads.svg)](https://anaconda.org/conda-forge/qpsolvers)

```console
conda install -c conda-forge qpsolvers
```

## Solvers

| Solver | Keyword | Algorithm | API | License | Warm-start |
| ------ | ------- | --------- | --- | ------- |------------|
| [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) | ``clarabel`` | Interior point | Sparse | Apache-2.0 | ✖️ |
| [CVXOPT](http://cvxopt.org/) | ``cvxopt`` | Interior point | Dense | GPL-3.0 | ✔️ |
| [DAQP](https://github.com/darnstrom/daqp) | ``daqp`` | Active set | Dense | MIT | ✖️ |
| [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) | ``ecos`` | Interior point | Sparse | GPL-3.0 | ✖️ |
| [Gurobi](https://www.gurobi.com/) | ``gurobi`` | Interior point | Sparse | Commercial | ✖️ |
| [HiGHS](https://highs.dev/) | ``highs`` | Active set | Sparse | MIT | ✖️ |
| [HPIPM](https://github.com/giaf/hpipm) | ``hpipm`` | Interior point | Dense | BSD-2-Clause | ✔️ |
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

- [Can I print the list of solvers available on my machine?](https://github.com/qpsolvers/qpsolvers/discussions/37)
- [Is it possible to solve a least squares rather than a quadratic program?](https://github.com/qpsolvers/qpsolvers/discussions/223)
- [I have a squared norm in my cost function, how can I apply a QP solver to my problem?](https://github.com/qpsolvers/qpsolvers/discussions/224)
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

The results below come from [`qpsolvers_benchmark`](https://github.com/qpsolvers/qpsolvers_benchmark), a benchmark for QP solvers in Python.

You can run the benchmark on your machine via a command-line tool (``pip install qpsolvers_benchmark``). Check out the benchmark repository for details. In the following tables, solvers are called with their default settings and compared over whole test sets by [shifted geometric mean](https://github.com/qpsolvers/qpsolvers_benchmark#shifted-geometric-mean) ("shm" for short; lower is better). We don't report the [GitHub free-for-all test set](https://github.com/qpsolvers/qpsolvers_benchmark/blob/main/github_ffa/results/github_ffa.md) yet, as it is still too small to be representative.

### Maros-Meszaros (hard problems)

Check out the [full report](https://github.com/qpsolvers/qpsolvers_benchmark/blob/7da937e0380ade8c109340bac4b4fe81f02e6806/maros_meszaros/results/maros_meszaros.md) for high- and low-accuracy solver settings.

|          |                    Success rate (%) |                        Runtime (shm) |                       Primal residual (shm) |                            Dual residual (shm) |                   Duality gap (shm) |                  Cost error (shm) |
|:---------|------------------------------------:|-------------------------------------:|--------------------------------------------:|----------------------------------------:|------------------------------------:|----------------------------------:|
| clarabel |                                89.9 |                                  1.0 |                                         1.0 |                                     1.9 |                                 1.0 |                               1.0 |
| cvxopt   |                                53.6 |                                 13.8 |                                         5.3 |                                     2.6 |                                22.9 |                               6.6 |
| gurobi   |                                16.7 |                                 57.8 |                                        10.5 |                                    37.5 |                                94.0 |                              34.9 |
| highs    |                                53.6 |                                 11.3 |                                         5.3 |                                     2.6 |                                21.2 |                               6.1 |
| osqp     |                                41.3 |                                  1.8 |                                        58.7 |                                    22.6 |                              1950.7 |                              42.4 |
| proxqp   |                                77.5 |                                  4.6 |                                         2.0 |                                     1.0 |                                11.5 |                               2.2 |
| scs      |                                60.1 |                                  2.1 |                                        37.5 |                                     3.4 |                               133.1 |                               8.4 |

### Maros-Meszaros dense (subset of dense problems)

Check out the [full report](https://github.com/qpsolvers/qpsolvers_benchmark/blob/7da937e0380ade8c109340bac4b4fe81f02e6806/maros_meszaros/results/maros_meszaros_dense.md) for high- and low-accuracy solver settings.

|          |                    Success rate (%) |                        Runtime (shm) |                       Primal residual (shm) |                            Dual residual (shm) |                   Duality gap (shm) |                  Cost error (shm) |
|:---------|------------------------------------:|-------------------------------------:|--------------------------------------------:|----------------------------------------:|------------------------------------:|----------------------------------:|
| clarabel |                               100.0 |                                  1.0 |                                         1.0 |                                    78.4 |                                 1.0 |                               1.0 |
| cvxopt   |                                66.1 |                               1267.4 |                                 292269757.0 |                                268292.6 |                               269.1 |                              72.5 |
| daqp     |                                50.0 |                               4163.4 |                                1056090169.5 |                                491187.7 |                               351.8 |                             280.0 |
| ecos     |                                12.9 |                              27499.0 |                                 996322577.2 |                                938191.8 |                               197.6 |                            1493.3 |
| gurobi   |                                37.1 |                               3511.4 |                                 497416073.4 |                              13585671.6 |                              4964.0 |                             190.6 |
| highs    |                                64.5 |                               1008.4 |                                 255341695.6 |                                235041.8 |                               396.2 |                              54.5 |
| osqp     |                                51.6 |                                371.7 |                                5481100037.5 |                               3631889.3 |                             24185.1 |                             618.4 |
| proxqp   |                                91.9 |                                 14.1 |                                      1184.3 |                                     1.0 |                                71.8 |                               7.2 |
| qpoases  |                                24.2 |                               3916.0 |                                8020840724.2 |                              23288184.8 |                               102.2 |                             778.7 |
| qpswift  |                                25.8 |                              16109.1 |                                 860033995.1 |                                789471.9 |                               170.4 |                             875.0 |
| quadprog |                                62.9 |                               1430.6 |                                 315885538.2 |                               4734021.7 |                              2200.0 |                             192.3 |
| scs      |                                72.6 |                                 95.6 |                                2817718628.1 |                                369300.9 |                              3303.2 |                             152.5 |

## Contributing

We welcome contributions, see the [contribution guidelines](https://github.com/qpsolvers/qpsolvers/blob/master/CONTRIBUTING.md) for details. We are also looking forward to hearing about your use cases! Please share them in [Show and tell](https://github.com/qpsolvers/qpsolvers/discussions/categories/show-and-tell).

## Citing qpsolvers

If you find this project useful, please consider giving it a :star: or citing it :books: A citation template is available via the ``Cite this repository`` button on GitHub.
