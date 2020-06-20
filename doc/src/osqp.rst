****
OSQP
****

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving convex quadratic programs in the form

.. math::

    \begin{split}\begin{array}{ll}
      \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
      \mbox{subject to} & l \leq A x \leq u
    \end{array}\end{split}

where :math:`x` is the optimization variable and :math:`P \in \mathbf{S}^n_+`
is a positive semidefinite matrix.

.. autofunction:: qpsolvers.osqp_solve_qp
