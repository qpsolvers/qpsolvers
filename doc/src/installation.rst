************
Installation
************

The simplest way to install this module is:

.. code:: bash

    pip install qpsolvers

You can add the ``--user`` parameter for a user-only installation.

Installing qpOASES
==================

Check out the `official qpOASES installation page
<https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation>`_ for
instructions. You might run into errors at the ``make python`` step. If so, you
can check out qpOASES from `this fork
<https://github.com/stephane-caron/qpOASES>`_, or check out its 4–5 last
commits to see how to cope with these issues.

Virtualenv
==========

To create an isolated environment using `Virtualenv <https://virtualenv.pypa.io>`_, you can do:

.. code:: bash

    virtualenv -p /usr/bin/python2.7 qpsolvers_env
    source qpsolvers_env/bin/activate
    cd qpsolvers_env
    git clone https://github.com/stephane-caron/qpsolvers
    cd qpsolvers
    pip install Cython numpy scipy
    pip install -r requirements.txt

Finally, you can run `deactivate` to exit virtualenv.
