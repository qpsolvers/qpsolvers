.. title:: Table of Contents

#########
qpsolvers
#########

.. **Release 1.3.1 -- June 13, 2020**

The goal of this library is to provide a unified interface to a wide array of
Quadratic Programming (QP) solvers available in Python.

The simplest way to install it is to run ``pip install qpsolvers``. You can add
the ``--user`` parameter for a user-only installation.

Usage
=====

The function ``solve_qp(P, q, G, h, A, b, lb, ub)`` is called with the
``solver`` keyword argument to select the backend solver. The quadratic program
it solves is, in standard form:

.. math::
    \begin{split}\begin{array}{ll}
      \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
      \mbox{subject to} & G x \leq h \\
        & A x = h \\
        & lb \leq x \leq ub
    \end{array}\end{split}

where :math:`x` is the optimization variable, and vector inequalities are taken
coordinate by coordinate.

Solvers
=======

Supported solvers currently include:

.. toctree::

    cvxopt.rst
    ecos.rst
    gurobi.rst
    mosek.rst
    osqp.rst
    qpoases.rst
    quadprog.rst

Example
=======

To solve a quadratic program, simply build the matrices that define it and call
the ``solve_qp`` function:

.. code:: python

    from numpy import array, dot
    from qpsolvers import solve_qp

    M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = dot(M.T, M)  # quick way to build a symmetric matrix
    q = dot(array([3., 2., 3.]), M).reshape((3,))
    G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))
    A = array([1., 1., 1.])
    b = array([1.])

    x = solve_qp(P, q, G, h, A, b)
    print("QP solution: x = {}".format(x))

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``.
