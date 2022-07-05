:github_url: https://github.com/stephane-caron/qpsolvers/tree/master/doc/src/quadratic-programming.rst

.. _Quadratic programming:

*********************
Quadratic programming
*********************

To solve a quadratic program, simply build the matrices that define it and call
the :func:`.solve_qp` function:

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

    x = solve_qp(P, q, G, h, A, b, solver="osqp")
    print(f"QP solution: x = {x}")

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``.
The backend QP solver used by :func:`.solve_qp` is selected via the ``solver``
keyword argument.

.. autofunction:: qpsolvers.solve_qp

Installed solvers are listed in:

.. autodata:: qpsolvers.available_solvers

See the ``examples/`` folder in the repository for other use cases. For a more
general introduction you can also check out this post on `quadratic programming
in Python <https://scaron.info/blog/quadratic-programming-in-python.html>`_.
