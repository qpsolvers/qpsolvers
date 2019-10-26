#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

classifiers = """\
Development Status :: 5 - Production/Stable
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Intended Audience :: Developers
Intended Audience :: Science/Research
Topic :: Scientific/Engineering :: Mathematics
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Operating System :: OS Independent"""

i = 'https://raw.githubusercontent.com/stephane-caron/qpsolvers/master/.qp.png'

long_description = """\
This module provides a single function ``solve_qp(P, q, G, h, A, b, solver=X)``
with a *solver* keyword argument to select the backend solver. The quadratic
program it solves is, in standard form:

    .. figure:: %s

where vector inequalities are taken coordinate by coordinate.

Solvers
-------

The list of supported solvers currently includes:

- Dense solvers:
    - `CVXOPT <http://cvxopt.org/>`_
    - `qpOASES <https://projects.coin-or.org/qpOASES>`_
    - `quadprog <https://pypi.python.org/pypi/quadprog>`_
- Sparse solvers:
    - `ECOS <https://www.embotech.com/ECOS>`_
      as wrapped by `CVXPY <http://www.cvxpy.org/>`_
    - `Gurobi <https://www.gurobi.com/>`_
    - `MOSEK <https://mosek.com/>`_
    - `OSQP <https://github.com/oxfordcontrol/osqp>`_

Example
-------

To solve a quadratic program, simply build the matrices that define it and call
the ``solve_qp`` function:

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

    print "QP solution:", solve_qp(P, q, G, h, A, b)

This example outputs the solution ``[0.30769231, -0.69230769,  1.38461538]``.
""" % i

setup(
    name='qpsolvers',
    version='1.0.7',
    description="Quadratic Programming solvers for Python with a unified API",
    long_description=long_description,
    url="https://github.com/stephane-caron/qpsolvers",
    author="Stéphane Caron",
    author_email="stephane.caron@normalesup.org",
    license="LGPL",
    keywords="qp, quadratic programming, solver",
    platforms="any",
    classifiers=classifiers.split('\n'),
    packages=['qpsolvers'],
    install_requires=['quadprog']
)
