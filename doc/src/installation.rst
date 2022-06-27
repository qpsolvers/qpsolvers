:github_url: https://github.com/stephane-caron/qpsolvers/tree/master/doc/src/installation.rst

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

Python 2
--------

If you have an older system with Python 2, for instance Ubuntu 16.04, try:

.. code:: bash

    sudo apt install python-dev
    pip qpsolvers==1.4.1

Python 2 is not supported any more, but this may still work. Note that
vulnerabilities `have been discovered
<https://github.com/stephane-caron/qpsolvers/pull/49>`_ in the dependencies of
this old version.

Windows
=======

- First, install the `Visual C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
- Install your Python environment, for instance `Anaconda <https://docs.anaconda.com/anaconda/install/windows/>`_
- Open a terminal configured for Python, for instance from the Anaconda Navigator, and run:

.. code:: bash

    pip install qpsolvers

Microsoft Visual Studio
-----------------------

- Open Microsoft Visual Studio
- Create a new project:
    - Select a new "Python Application" project template
    - Click "Next"
    - Give a name to your project
    - Click "Create"
- Go to Tools → Python → Python Environments:
    - To the left of the "Python Environments" tab that opens, select a Python version >= 3.8
    - Click on "Packages (PyPI)"
    - In the search box, type "qpsolvers"
    - Below the search box, click on "Run command: pip install qpsolvers"
    - A window pops up asking for administrator privileges: grant them
    - Check the text messages in the "Output" pane at the bottom of the window
- Go to the main code tab (it should be your project name followed by the ".py" extension)
- Copy the `example code <https://github.com/stephane-caron/qpsolvers#example>`_ from the README and paste it there
- Click on the "Run" icon in the toolbar to execute this program

At this point a ``python.exe`` window should open with the following output:

.. code:: bash

    QP solution: x = [0.30769231, -0.69230769, 1.38461538]
    Press any key to continue . . .

Solvers
=======

Open source solvers
-------------------

To install at once all QP solvers available from the `Python Package Index
<https://pypi.org/>`_, run the ``pip`` command with the optional
``all_pypi_solvers`` dependency:

.. code:: bash

    pip3 install "qpsolvers[all_pypi_solvers]"

This may take a while.

.. _gurobi-install:

Gurobi
------

Gurobi comes with a `one-line pip installation
<https://www.gurobi.com/documentation/9.1/quickstart_linux/cs_using_pip_to_install_gr.html>`_
where you can fetch the solver directly from the company servers:

.. code:: bash

    python -m pip install -i https://pypi.gurobi.com gurobipy

This version comes with limitations. For instance, trying to solve a problem
with 200 optimization variables fails with the following warning:

.. code:: python

    Warning: Model too large for size-limited license; visit https://www.gurobi.com/free-trial for a full license

.. _qpoases-install:

quadprog
--------

If you have a C compiler, you can install the quadprog solver from source:

.. code:: bash

    pip install quadprog

If this command returns a `build error
<https://github.com/quadprog/quadprog/issues/15>`__, you can install the solver
from pre-built wheels instead:

.. code:: bash

    pip install quadprog-wheel

qpOASES
-------

Check out the `official qpOASES installation page
<https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation>`_ for the
latest release. However, you might run into errors at the ``make python`` step.
If so, you can check out qpOASES from `this fork
<https://github.com/stephane-caron/qpOASES>`_ and follow these instructions:

.. code:: bash

    git clone --recursive https://github.com/stephane-caron/qpOASES.git
    cd qpOASES
    make
    cd interfaces/python
    python setup.py install --user

The `setup.py` script takes the same command-line arguments as `pip`. Remove
`--user` and run it as root to install the library system-wide.
