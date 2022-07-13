:github_url: https://github.com/stephane-caron/qpsolvers/tree/master/doc/src/developer-notes.rst

***************
Developer notes
***************

Adding a new solver
===================

Let's imagine we want to add a new solver called *AwesomeQP*. The solver keyword, the string passed via the ``solver`` keyword argument, is the lowercase version of the vernacular name of a QP solver. For our imaginary solver, the keyword is therefore ``"awesomeqp"``.

The process to add AwesomeQP to *qpsolvers* goes as follows:

1. Create a new file ``qpsolvers/solvers/awesomeqp_.py`` (named after the solver keyword, with a trailing underscore)
2. Define in this file a function ``awesomeqp_solve_qp``
3. Define the function prototype for ``awesomeqp_solve_qp`` in ``qpsolvers/solvers/__init__.py``:

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

4. Below the prototype, import the function into the ``solve_function`` dictionary:

.. code:: python

    try:
        from .awesomeqp_ import awesomeqp_solve_qp

        solve_function["awesomeqp"] = awesomeqp_solve_qp
        available_solvers.append("awesomeqp")
        # dense_solvers.append("awesomeqp")   if applicable
        # sparse_solvers.append("awesomeqp")  if applicable
    except ImportError:
        pass

5. Append the solver identifier to ``dense_solvers`` or ``sparse_solvers``, if applicable
6. Import ``awesomeqp_solve_qp`` from ``qpsolvers/__init__.py`` and add it to ``__all__``
7. Import ``awesomeqp_solve_qp`` from the convenience ``__init__.py`` at the root of the repository, and add it to ``__all__``
8. Update the benchmark: run the scripts ``examples/benchmark_*.py`` and update the corresponding tables and plot in the README

Testing locally
===============

To run all CI checks locally, go to the repository folder and run

.. code:: bash

    tox -e py

This will run linters and unit tests.
