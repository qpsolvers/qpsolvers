# QP Solvers

Wrapper around Quadratic Programming (QP) solvers in Python with a unified API.

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
    - [MOSEK](https://mosek.com/)\*
    - [qpOASES](https://projects.coin-or.org/qpOASES)
    - [quadprog](https://pypi.python.org/pypi/quadprog/)
- Sparse solvers:
    - [CVXPY](http://www.cvxpy.org/)
    - [Gurobi](https://www.gurobi.com/)
    - [OSQP](https://github.com/oxfordcontrol/osqp)

\* MOSEK is called via CVXOPT and therefore treated as a dense solver, although
it performs best as a sparse one.

## Performances

On the [dense.py](examples/dense.py) distributed in the examples folder, the
performance of all solvers (as measured by IPython's ``%timeit`` on my machine)
is:

| Solver   | Type   | Time (ms) |
| -------- | ------ | --------- |
| qpoases  | Dense  | 0.03      |
| quadprog | Dense  | 0.03      |
| cvxopt   | Dense  | 0.55      |
| osqp     | Sparse | 0.36      |
| gurobi   | Sparse | 0.86      |
| cvxpy    | Sparse | 2.81      |
| mosek    | Sparse | 7.24      |

Meanwhile, on the [sparse.py](examples/sparse.py) example, these performances
become:

| Solver   | Type   | Time (s) |
| -------- | ------ | -------- |
| qpoases  | Dense  | 15.7     |
| quadprog | Dense  | 3.8      |
| cvxopt   | Dense  | 1.5      |
| gurobi   | Sparse | 0.5      |
| cvxpy    | Sparse | 0.6      |
| mosek    | Sparse | 0.01     |

Finally, here are the results on a benchmark of random problems generated with
the [randomized.py](examples/randomized.py) example (each data point
corresponds to an average over 10 runs):

<img src="https://scaron.info/images/qp-benchmark.png">

Note that performances of QP solvers largely depend on the problem solved. This
benchmark only considers dense problems, hence the clear advantage of dense
solvers over sparse ones.
