# Ô QP

Wrapper around Quadratic Programming (QP) solvers in Python with a unified API.

## Usage

A ``solve_qp_X(P, q[, G, h[, A, b[, initvals]])`` function is provided for each
available solver ``X``. The format is the same as
[cvxopt.solvers.qp](http://cvxopt.org/userguide/coneprog.html#quadratic-programming),
the function solves the quadratic program:

<img
src="http://cvxopt.org/userguide/_images/math/305efdce8b67069139cfdce108379dd0f9c13e14.png">

## Solvers

The list of supported solvers currently includes:

- Numerical solvers:
    - [CVXOPT](http://cvxopt.org/)
    - [qpOASES](https://projects.coin-or.org/qpOASES)
    - [quadprog](https://pypi.python.org/pypi/quadprog/)
- Symbolic solvers:
    - [CVXPY](http://www.cvxpy.org/en/latest/)
    - [Gurobi](http://www.gurobi.com/)

Numerical solvers are those for which the input can be provided directly in
matrix-vector form. On the contrary, symbolic solvers require calls to various
constructors, so that building their input usually takes more time than solving
the QP itself.
