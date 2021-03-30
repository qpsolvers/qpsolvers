***************
Getting started
***************

Quadratic programming
=====================

The function ``solve_qp(P, q, G, h, A, b, lb, ub)`` is called with the
``solver`` keyword argument to select the backend solver:

.. autofunction:: qpsolvers.solve_qp

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

See the ``examples/`` folder in the repository for more advanced use cases.

Least squares
=============

The function ``solve_ls(P, q, G, h, A, b, lb, ub)`` is called with the
``solver`` keyword argument to select the backend solver:

.. autofunction:: qpsolvers.solve_ls

To solve a linear least-squares problem, simply build the matrices that define
it and call the ``solve_ls`` function:

.. code:: python

    from numpy import array, dot
    from qpsolvers import solve_ls

    R = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    s = array([3., 2., 3.])
    G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))

    x_sol = solve_ls(R, s, G, h, solver=solver, verbose=True)
    print("LS solution: x = {}".format(x))

This example outputs the solution ``[-0.0530504, 0.0265252, 2.1061008]``.

See the ``examples/`` folder in the repository for more advanced use cases.
