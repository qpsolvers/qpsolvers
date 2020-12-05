************
Installation
************

Linux
=====

The simplest way to install the package on a recent Debian-based system with
Python 3 is:

.. code:: bash

    sudo apt install python3-dev
    pip3 install qpsolvers

You can add the ``--user`` parameter for a user-only installation.

Windows
=======

- First, install the `Visual C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
- Install your Python environment, for instance `Anaconda <https://docs.anaconda.com/anaconda/install/windows/>`_
- Open a terminal configured for Python, for instance from the Anaconda Navigator, and run:

.. code:: bash

    pip install qpsolvers

Python 2
========

If you have an older system with Python 2, for instance Ubuntu 16.04, try:

.. code:: bash

    sudo apt install python-dev
    pip install -r requirements2.txt

You can also add the ``--user`` parameter for a user-only installation.

Virtual Environment
-------------------

To create an isolated environment using `Virtualenv <https://virtualenv.pypa.io>`_, you can do:

.. code:: bash

    virtualenv -p /usr/bin/python2.7 qpsolvers_env
    source qpsolvers_env/bin/activate
    cd qpsolvers_env
    git clone https://github.com/stephane-caron/qpsolvers
    cd qpsolvers
    pip install Cython numpy scipy
    pip install -r requirements2.txt

Finally, you can run `deactivate` to exit virtualenv.

Solvers
=======

qpOASES
-------

Check out the `official qpOASES installation page
<https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation>`_ for
instructions. You might run into errors at the ``make python`` step. If so, you
can check out qpOASES from `this fork
<https://github.com/stephane-caron/qpOASES>`_, or check out its 4–5 last
commits to see how to cope with these issues.
