# QP Solvers

Wrapper around Quadratic Programming (QP) solvers in Python with a unified API.

## Usage

The function ``solve_qp(P, q, G, h)`` is called with the ``solver='xyz'``
keyword argument to select the backend solver. Matrix names are the same as in
[cvxopt.solvers.qp](http://cvxopt.org/userguide/coneprog.html#quadratic-programming),
the function solves the quadratic program:

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

First, note that performances of QP solvers largely depend on the problem
solved. For example, active-set solvers (e.g. qpOASES and quadprog) are usually
faster on smaller problems, while interior-point methods (e.g. MOSEK) scale
better. On the [small.py](examples/small.py) distributed in the examples folder,
the performance of all solvers (as mesured by IPython's ``%timeit`` on my
machine) is:

| Solver   | Time (µs) |
| -------- | ----------|
| qpoases  | 31.5      |
| quadprog | 34.1      |
| cvxopt   | 559       |
| gurobi   | 865       |
| cvxpy    | 2.810     |
| mosek    | 7.240     |
