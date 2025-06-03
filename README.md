# Quadratic Programming Solvers in Python

[![CI](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/ci.yml?branch=main)](https://github.com/qpsolvers/qpsolvers/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/qpsolvers/qpsolvers/docs.yml?branch=main&label=docs)](https://qpsolvers.github.io/qpsolvers/)
[![Coverage](https://coveralls.io/repos/github/qpsolvers/qpsolvers/badge.svg?branch=main)](https://coveralls.io/github/qpsolvers/qpsolvers?branch=main)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/qpsolvers.svg?color=blue)](https://anaconda.org/conda-forge/qpsolvers)
[![PyPI version](https://img.shields.io/pypi/v/qpsolvers?color=blue)](https://pypi.org/project/qpsolvers/)
[![PyPI downloads](https://img.shields.io/pypi/dm/qpsolvers?color=blue)](https://pypistats.org/packages/qpsolvers)

This library provides a [`solve_qp`](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp) function to solve convex quadratic programs:

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
print(f"QP solution: {x = }")
```

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``. It is also possible to get dual multipliers at the solution, as shown in [this example](https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#dual-multipliers).

## Installation

### From conda-forge

```console
conda install -c conda-forge qpsolvers
```

### From PyPI

To install the library with open source QP solvers:

```console
pip install qpsolvers[open_source_solvers]
```

This one-size-fits-all installation may not work immediately on all systems (for instance if [a solver tries to compile from source](https://github.com/quadprog/quadprog/issues/42)). If you run into any issue, check out the following variants:

- ``pip install qpsolvers[wheels_only]`` will only install solvers with pre-compiled binaries,
- ``pip install qpsolvers[clarabel,daqp,proxqp,scs]`` (for instance) will install the listed set of QP solvers,
- ``pip install qpsolvers`` will only install the library itself.

When imported, qpsolvers loads all the solvers it can find and lists them in ``qpsolvers.available_solvers``.

## Solvers

| Solver | Keyword | Algorithm | API | License |
| ------ | ------- | --------- | --- | ------- |
| [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) | ``clarabel`` | Interior point | Sparse | Apache-2.0 |
| [CVXOPT](http://cvxopt.org/) | ``cvxopt`` | Interior point | Dense | GPL-3.0 |
| [DAQP](https://github.com/darnstrom/daqp) | ``daqp`` | Active set | Dense | MIT |
| [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) | ``ecos`` | Interior point | Sparse | GPL-3.0 |
| [Gurobi](https://www.gurobi.com/) | ``gurobi`` | Interior point | Sparse | Commercial |
| [HiGHS](https://highs.dev/) | ``highs`` | Active set | Sparse | MIT |
| [HPIPM](https://github.com/giaf/hpipm) | ``hpipm`` | Interior point | Dense | BSD-2-Clause |
| [jaxopt.OSQP](https://jaxopt.github.io/stable/_autosummary/jaxopt.OSQP.html) | ``jaxopt_osqp`` | Augmented Lagrangian | Dense | Apache-2.0 |
| [KVXOPT](https://github.com/sanurielf/kvxopt) | ``kvxopt`` | Interior point | Dense & Sparse | GPL-3.0 |
| [MOSEK](https://mosek.com/) | ``mosek`` | Interior point | Sparse | Commercial |
| NPPro | ``nppro`` | Active set | Dense | Commercial |
| [OSQP](https://osqp.org/) | ``osqp`` | Augmented Lagrangian | Sparse | Apache-2.0 |
| [PIQP](https://github.com/PREDICT-EPFL/piqp) | ``piqp`` | Proximal interior point | Dense & Sparse | BSD-2-Clause |
| [ProxQP](https://github.com/Simple-Robotics/proxsuite) | ``proxqp`` | Augmented Lagrangian | Dense & Sparse | BSD-2-Clause |
| [QPALM](https://github.com/kul-optec/QPALM) | ``qpalm`` | Augmented Lagrangian | Sparse | LGPL-3.0 |
| [qpax](https://github.com/kevin-tracy/qpax/) | ``qpax`` | Interior point | Dense | MIT |
| [qpOASES](https://github.com/coin-or/qpOASES) | ``qpoases`` | Active set | Dense | LGPL-2.1 |
| [qpSWIFT](https://github.com/qpSWIFT/qpSWIFT) | ``qpswift`` | Interior point | Sparse | GPL-3.0 |
| [quadprog](https://github.com/quadprog/quadprog) | ``quadprog`` | Active set | Dense | GPL-2.0 |
| [SCS](https://www.cvxgrp.org/scs/) | ``scs`` | Augmented Lagrangian | Sparse | MIT |
| [SIP](https://github.com/joaospinto/sip_python) | ``sip`` | Barrier Augmented Lagrangian | Sparse | MIT |

Matrix arguments are NumPy arrays for dense solvers and SciPy Compressed Sparse Column (CSC) matrices for sparse ones.

## Frequently Asked Questions

- [Can I print the list of solvers available on my machine?](https://github.com/qpsolvers/qpsolvers/discussions/37)
- [Is it possible to solve a least squares rather than a quadratic program?](https://github.com/qpsolvers/qpsolvers/discussions/223)
- [I have a squared norm in my cost function, how can I apply a QP solver to my problem?](https://github.com/qpsolvers/qpsolvers/discussions/224)
- [I have a non-convex quadratic program, is there a solver I can use?](https://github.com/qpsolvers/qpsolvers/discussions/240)
- [I have quadratic equality constraints, is there a solver I can use?](https://github.com/qpsolvers/qpsolvers/discussions/241)
- [Error: Mircrosoft Visual C++ 14.0 or greater is required on Windows](https://github.com/qpsolvers/qpsolvers/discussions/257)
- [Can I add penalty terms as in ridge regression or LASSO?](https://github.com/qpsolvers/qpsolvers/discussions/272)

## Benchmark

QP solvers come with their strengths and weaknesses depending on the algorithmic choices they make. To help you find the ones most suited to your problems, you can check out the results from [`qpbenchmark`](https://github.com/qpsolvers/qpbenchmark), a benchmark for QP solvers in Python. The benchmark is divided into test sets, each test set representing a different distribution of quadratic programs with specific dimensions and structure (large sparse problems, optimal control problems, ...):

- 📈 [Free-for-all test set](https://github.com/qpsolvers/free_for_all_qpbenchmark): open to all problems submitted by the community.
- 📈 [Maros-Meszaros test set](https://github.com/qpsolvers/maros_meszaros_qpbenchmark): hard problems curated by the numerical optimization community.
- 📈 [MPC test set](https://github.com/qpsolvers/mpc_qpbenchmark): convex model predictive control problems arising in robotics.

## Citing qpsolvers

If you find this project useful, please consider giving it a :star: or citing it if your work is scientific:

```bibtex
@software{qpsolvers,
  title = {{qpsolvers: Quadratic Programming Solvers in Python}},
  author = {Caron, Stéphane and Arnström, Daniel and Bonagiri, Suraj and Dechaume, Antoine and Flowers, Nikolai and Heins, Adam and Ishikawa, Takuma and Kenefake, Dustin and Mazzamuto, Giacomo and Meoli, Donato and O'Donoghue, Brendan and Oppenheimer, Adam A. and Pandala, Abhishek and Quiroz Omaña, Juan José and Rontsis, Nikitas and Shah, Paarth and St-Jean, Samuel and Vitucci, Nicola and Wolfers, Soeren and Yang, Fengyu and @bdelhaisse and @MeindertHH and @rimaddo and @urob and @shaoanlu and Khalil, Ahmed and Kozlov, Lev and Groudiev, Antoine and Sousa Pinto, João},
  license = {LGPL-3.0},
  url = {https://github.com/qpsolvers/qpsolvers},
  version = {4.7.1},
  year = {2025}
}
```

Don't forget to add yourself to the BibTeX above and to `CITATION.cff` if you contribute to this repository.

## Contributing

We welcome contributions! The first step is to install the library and use it. Report any bug in the [issue tracker](https://github.com/qpsolvers/qpsolvers/issues). If you're a developer looking to hack on open source, check out the [contribution guidelines](https://github.com/qpsolvers/qpsolvers/blob/main/CONTRIBUTING.md) for suggestions.

## See also

- [qpbenchmark](https://github.com/qpsolvers/qpbenchmark/): Benchmark for quadratic programming solvers available in Python.
- [qpsolvers-eigen](https://github.com/ami-iit/qpsolvers-eigen): C++ abstraction layer for quadratic programming solvers using Eigen.
