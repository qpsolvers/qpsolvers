***************
Developer notes
***************

Adding a new solver
===================

Let's imagine we want to add a new solver called *AwesomeQP*. The solver keyword, the string passed via the ``solver`` keyword argument, is the lowercase version of the vernacular name of a QP solver. For our imaginary solver, the keyword is therefore ``"awesomeqp"``.

The process to add AwesomeQP to *qpsolvers* goes as follows:

1. Create a new file ``qpsolvers/solvers/awesomeqp_.py`` (named after the solver keyword, with a trailing underscore)
2. Implement in this file a function ``awesomeqp_solve_problem`` that returns a :class:`.Solution`
3. Implement in the same file a function ``awesomeqp_solve_qp`` to connect it to the historical API, typically as follows:

.. code:: python

    def awesomeqp_solve_qp(P, q, G, h, A, b, lb, ub, initvals=None, verbose=False, **kwargs):
    ) -> Optional[np.ndarray]:
        r"""Solve a quadratic program using AwesomeQP.

        [document parameters and return values here]
        """
        problem = Problem(P, q, G, h, A, b, lb, ub)
        solution = awesomeqp_solve_problem(
            problem, initvals, verbose, backend, **kwargs
        )
        return solution.x if solution.found else None

4. Define the two function prototypes for ``awesomeqp_solve_problem`` and ``awesomeqp_solve_qp`` in ``qpsolvers/solvers/__init__.py``:

.. code:: python

    # AwesomeQP
    # ========

    awesome_solve_qp: Optional[
        Callable[
            [
                ndarray,
                ndarray,
                Optional[ndarray],
                Optional[ndarray],
                Optional[ndarray],
                Optional[ndarray],
                Optional[ndarray],
                Optional[str],
                bool,
            ],
            Optional[ndarray],
        ]
    ] = None

.. note::

    The prototype needs to match the actual function. You can check its correctness by running ``tox -e py`` in the repository.

5. Below the prototype, import the function into the ``solve_function`` dictionary:

.. code:: python

    try:
        from .awesomeqp_ import awesomeqp_solve_qp

        solve_function["awesomeqp"] = awesomeqp_solve_qp
        available_solvers.append("awesomeqp")
        # dense_solvers.append("awesomeqp")   if applicable
        # sparse_solvers.append("awesomeqp")  if applicable
    except ImportError:
        pass

6. Append the solver identifier to ``dense_solvers`` or ``sparse_solvers``, if applicable
7. Import ``awesomeqp_solve_qp`` from ``qpsolvers/__init__.py`` and add it to ``__all__``
8. Add the solver to ``doc/src/supported-solvers.rst``
9. Add the solver to the *Solvers* section of the README
10. Assuming AwesomeQP is distributed on `PyPI <https://pypi.org/>`__, add it to the ``[testenv]`` and ``[testenv:coverage]`` environments of ``tox.ini`` for unit testing
11. Assuming AwesomeQP is distributed on Conda or PyPI, add it to the list of dependencies in ``doc/environment.yml``
12. Log the new solver as an addition in the changelog
13. If you are a new contributor, feel free to add your name to ``CITATION.cff``.

Problem conversions
===================

.. automodule:: qpsolvers.conversions
    :members:

Testing locally
===============

To run all CI checks locally, go to the repository folder and run

.. code:: bash

    tox -e py

This will run linters and unit tests.
