# QP Solvers for Python

Wrapper around Quadratic Programming (QP) solvers in Python, with a unified
interface.

## Installation

The simplest way to install this module is:
```
pip install qpsolvers
```
You can add the ``--user`` parameter for a user-only installation.

## Usage

The function ``solve_qp(P, q, G, h, A, b)`` is called with the ``solver``
keyword argument to select the backend solver. The quadratic program it solves
is, in standard form:

<img src=".qp.png">

Vector inequalities are taken coordinate by coordinate.

## Solvers

The list of supported solvers currently includes:

- Dense solvers:
    - [CVXOPT](http://cvxopt.org/)
    - [qpOASES](https://projects.coin-or.org/qpOASES)
    - [quadprog](https://pypi.python.org/pypi/quadprog/)
- Sparse solvers:
    - [ECOS](https://www.embotech.com/ECOS) as wrapped by [CVXPY](http://www.cvxpy.org/)
    - [Gurobi](https://www.gurobi.com/)
    - [MOSEK](https://mosek.com/)
    - [OSQP](https://github.com/oxfordcontrol/osqp)

## Performances

On the [dense.py](examples/dense.py) distributed in the examples folder, the
performance of all solvers (as measured by IPython's ``%timeit`` on my machine)
is:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| quadprog | Dense  | 0.02      |
| qpoases  | Dense  | 0.03      |
| osqp     | Sparse | 0.04      |
| cvxopt   | Dense  | 0.43      |
| gurobi   | Sparse | 0.84      |
| ecos     | Sparse | 2.61      |
| mosek    | Sparse | 7.17      |

Meanwhile, on the [sparse.py](examples/sparse.py) example, these performances
become:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| osqp     | Sparse |    1      |
| mosek    | Sparse |   17      |
| cvxopt   | Dense  |   35      |
| gurobi   | Sparse |  221      |
| quadprog | Dense  |  421      |
| ecos     | Sparse |  638      |
| qpoases  | Dense  | 2210      |

Finally, here are the results on a benchmark of random problems generated with
the [randomized.py](examples/randomized.py) example (each data point
corresponds to an average over 10 runs):

<img src="https://scaron.info/images/qp-benchmark.png">

Note that performances of QP solvers largely depend on the problem solved. For
instance, MOSEK performs an [automatic conversion to Second-Order Cone
Programming
(SOCP)](https://docs.mosek.com/8.1/pythonapi/prob-def-quadratic.html) which the
documentation advises bypassing for better performance. Similarly, ECOS
reformulates from QP to SOCP and [works best on small
problems](https://web.stanford.edu/%7Eboyd/papers/ecos.html).
