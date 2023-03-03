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

    x_sol = solve_ls(R, s, G, h, solver="osqp")
    print(f"LS solution: x = {x}")

The backend QP solver is selected among :ref:`supported solvers <Supported
solvers>` via the ``solver`` keyword argument. This example outputs the
solution ``[-0.0530504, 0.0265252, 2.1061008]``.

.. autofunction:: qpsolvers.solve_ls

See the ``examples/`` folder in the repository for more advanced use cases. For
a more general introduction you can also check out this post on `least squares
in Python <https://scaron.info/robotics/least-squares.html>`_.
