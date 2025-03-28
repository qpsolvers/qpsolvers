.. _Quadratic programming:

*********************
Quadratic programming
*********************

Primal problem
==============

A quadratic program is defined in standard form as:

.. math::

    \begin{split}\begin{array}{ll}
        \underset{x}{\mbox{minimize}} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
    \end{array}\end{split}

The vectors :math:`lb` and :math:`ub` can contain :math:`\pm \infty` values to
disable bounds on some coordinates. To solve such a problem, build the matrices
that define it and call the :func:`.solve_qp` function:

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

The backend QP solver is selected among :ref:`supported solvers <Supported
solvers>` via the ``solver`` keyword argument. This example outputs the
solution ``[0.30769231, -0.69230769,  1.38461538]``.

.. autofunction:: qpsolvers.solve_qp

See the ``examples/`` folder in the repository for more use cases. For a more
general introduction you can also check out this post on `quadratic programming
in Python <https://scaron.info/blog/quadratic-programming-in-python.html>`_.

Problem class
=============

Alternatively, we can define the matrices and vectors using the :class:`.Problem` class:

.. autoclass:: qpsolvers.problem.Problem
   :members:

The solve function corresponding to :class:`.Problem` is :func:`.solve_problem`
rather than :func:`.solve_qp`.

Dual multipliers
================

The dual of the quadratic program defined above can be written as:

.. math::

    \begin{split}\begin{array}{ll}
    \underset{x, z, y, z_{\mathit{box}}}{\mbox{maximize}} &
        -\frac{1}{2} x^T P x - h^T z - b^T y
        - lb^T z_{\mathit{box}}^- - ub^T z_{\mathit{box}}^+ \\
    \mbox{subject to}
        & P x + G^T z + A^T y + z_{\mathit{box}} + q = 0 \\
        & z \geq 0
    \end{array}\end{split}

were :math:`v^- = \min(v, 0)` and :math:`v^+ = \max(v, 0)`. To solve both a
problem and its dual, getting a full primal-dual solution :math:`(x^*, z^*,
y^*, z_\mathit{box}^*)`, build a :class:`.Problem` and call the
:func:`.solve_problem` function:

.. code:: python

    import numpy as np
    from qpsolvers import Problem, solve_problem

    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = M.T.dot(M)  # quick way to build a symmetric matrix
    q = np.array([3., 2., 3.]).dot(M).reshape((3,))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))
    A = np.array([1., 1., 1.])
    b = np.array([1.])
    lb = -0.6 * np.ones(3)
    ub = +0.7 * np.ones(3)

    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = solve_problem(problem, solver="proxqp")

    print(f"Primal: x = {solution.x}")
    print(f"Dual (Gx <= h): z = {solution.z}")
    print(f"Dual (Ax == b): y = {solution.y}")
    print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")

The function returns a :class:`.Solution` with both primal and dual vectors. This example outputs the following solution:

.. code::

    Primal: x = [ 0.63333169 -0.33333307  0.70000137]
    Dual (Gx <= h): z = [0.         0.         7.66660538]
    Dual (Ax == b): y = [-16.63326017]
    Dual (lb <= x <= ub): z_box = [ 0.          0.         26.26649724]

.. autofunction:: qpsolvers.solve_problem

See the ``examples/`` folder in the repository for more use cases. For an
introduction to dual multipliers you can also check out this post on
`optimality conditions and numerical tolerances in QP solvers
<https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html>`_.

Optimality of a solution
========================

The :class:`.Solution` class describes the solution found by a solver to a
given problem. It is linked to the corresponding :class:`.Problem`, which it
can use for instance to check residuals. We can for instance check the
optimality of the solution returned by a solver with:

.. code:: python

    import numpy as np
    from qpsolvers import Problem, solve_problem

    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = M.T.dot(M)  # quick way to build a symmetric matrix
    q = np.array([3., 2., 3.]).dot(M).reshape((3,))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))
    A = np.array([1., 1., 1.])
    b = np.array([1.])
    lb = -0.6 * np.ones(3)
    ub = +0.7 * np.ones(3)

    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = solve_problem(problem, solver="qpalm")

    print(f"- Solution is{'' if solution.is_optimal(1e-8) else ' NOT'} optimal")
    print(f"- Primal residual: {solution.primal_residual():.1e}")
    print(f"- Dual residual: {solution.dual_residual():.1e}")
    print(f"- Duality gap: {solution.duality_gap():.1e}")

This example prints:

.. code::

    - Solution is optimal
    - Primal residual: 1.1e-16
    - Dual residual: 1.4e-14
    - Duality gap: 0.0e+00

You can check out [Caron2022]_ for an overview of optimality conditions and why
a solution is optimal if and only if these three residuals are zero.

.. autoclass:: qpsolvers.solution.Solution
   :members:
