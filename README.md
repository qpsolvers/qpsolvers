# QP Solvers

Wrapper around Quadratic Programming (QP) solvers in Python with a unified API.

## Usage

The function ``solve_qp(P, q, G, h, A, b)`` is called with the ``solver``
keyword argument to select the backend solver. The quadratic program solved is,
in standard form:

<img src=".qp.png">

Vector inequalities are taken coordinate by coordinate. Check out the
``examples/`` folder to see practical calls.

## Solvers

The list of supported solvers currently includes:

- Numerical solvers:
    - [CVXOPT](http://cvxopt.org/)
    - [qpOASES](https://projects.coin-or.org/qpOASES)
    - [quadprog](https://pypi.python.org/pypi/quadprog/)
- Symbolic solvers:
    - [CVXPY](http://www.cvxpy.org/en/latest/)
    - [Gurobi](https://www.gurobi.com/)
    - [MOSEK](https://mosek.com/) as wrapped by CVXOPT

Numerical solvers are those for which the input can be provided directly in
matrix-vector form. On the contrary, symbolic solvers call various
constructors, so that building their input usually takes more time than solving
the QP itself.

## Performances

On the [small.py](examples/small.py) distributed in the examples folder, the
performance of all solvers (as mesured by IPython's ``%timeit`` on my machine)
is:

| Solver   | Time (µs) |
| -------- | ----------|
| qpoases  | 31.5      |
| quadprog | 34.1      |
| cvxopt   | 559       |
| gurobi   | 865       |
| cvxpy    | 2.810     |
| mosek    | 7.240     |

Here are the results on a benchmark of random problems generated with the
[randomized.py](examples/randomized.py) example (each data point corresponds to
an average over 10 runs):

<img src="https://scaron.info/images/qp-benchmark.png">

Note that performances of QP solvers largely depend on the problem solved. In
this benchmark, I only considered dense problems, hence the clear advantage of
active-set solvers (quadprog or qpOASES) over symbolic solvers (CVXPY or
Gurobi).
