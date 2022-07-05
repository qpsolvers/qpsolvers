:github_url: https://github.com/stephane-caron/qpsolvers/tree/master/doc/src/least-squares.rst

.. _Least squares:

*************
Least squares
*************

To solve a linear least-squares problem, simply build the matrices that define
it and call the :func:`.solve_ls` function:

.. code:: python

    from numpy import array, dot
    from qpsolvers import solve_ls

    R = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    s = array([3., 2., 3.])
    G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))

    x_sol = solve_ls(R, s, G, h, solver="osqp", verbose=True)
    print(f"LS solution: x = {x}")

This example outputs the solution ``[-0.0530504, 0.0265252, 2.1061008]``. The
backend QP solver used by :func:`.solve_ls` is selected via the ``solver``
keyword argument.

.. autofunction:: qpsolvers.solve_ls

See the ``examples/`` folder in the repository for more advanced use cases. For
a more general introduction you can also check out this post on `least squares
in Python <https://scaron.info/robotics/least-squares.html>`_.
