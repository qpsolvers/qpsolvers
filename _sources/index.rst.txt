.. title:: Table of Contents

#########
qpsolvers
#########

Unified interface to Quadratic Programming (QP) solvers available in Python.

The library provides a one-stop shop :func:`.solve_qp` function with a ``solver`` keyword argument to select the backend solver. It solves :ref:`convex quadratic programs <Quadratic programming>` in standard form:

.. math::

    \begin{split}\begin{array}{ll}
        \underset{x}{\mbox{minimize}} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                    \\
            & lb \leq x \leq ub
    \end{array}\end{split}

A similar function is provided for :ref:`least squares <Least squares>`.

.. toctree::
    :maxdepth: 1

    installation.rst
    quadratic-programming.rst
    least-squares.rst
    supported-solvers.rst
    unsupported-solvers.rst
    developer-notes.rst
    references.rst
