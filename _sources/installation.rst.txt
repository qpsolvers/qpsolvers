************
Installation
************

Linux
=====

Conda
-----

To install the library from `conda-forge <https://conda-forge.org/>`__, simply run:

.. code:: bash

    conda install -c conda-forge qpsolvers

PyPI
----

First, install the pip package manager, for example on a recent Debian-based distribution with Python 3:

.. code:: bash

    sudo apt install python3-dev

You can then install the library by:

.. code:: bash

    pip install qpsolvers

Add the ``--user`` parameter for a user-only installation.

Windows
=======

Anaconda
--------

- First, install the `Visual C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
- Install your Python environment, for instance `Anaconda <https://docs.anaconda.com/anaconda/install/windows/>`_
- Install the library from conda-forge, for instance in a terminal opened from the Anaconda Navigator:

.. code:: bash

    conda install -c conda-forge qpsolvers

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
- Copy the `example code <https://github.com/qpsolvers/qpsolvers#example>`_ from the README and paste it there
- Click on the "Run" icon in the toolbar to execute this program

At this point a ``python.exe`` window should open with the following output:

.. code:: bash

    QP solution: x = [0.30769231, -0.69230769, 1.38461538]
    Press any key to continue . . .

Solvers
=======

Open source solvers
-------------------

To install at once all open source QP solvers available from the `Python
Package Index <https://pypi.org/>`_, run the ``pip`` command as follows:

.. code:: bash

    pip install "qpsolvers[open_source_solvers]"

You can also install a subset of QP solvers of your liking, for instance:

.. code:: bash

    pip install qpsolvers[clarabel,daqp,proxqp,scs]

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

HiGHS
-----

The simplest way to install HiGHS is:

.. code:: bash

    pip install highspy

If this solution doesn't work for you, follow the `Python installation
instructions <https://github.com/ERGO-Code/HiGHS#python>`__ from the README.

quadprog
--------

You can install the quadprog solver from PyPI:

.. code:: bash

    pip install quadprog

This package comes with wheels to avoid recompiling the solver from source.

qpOASES
-------

The simplest way to install qpOASES is via conda-forge:

.. code:: bash

    conda install -c conda-forge qpoases

You can also check out the `official qpOASES installation page
<https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation>`_ for the
latest release.
