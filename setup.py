#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup

setup(
    name='qpsolvers',
    version='0.1.0',
    description="Wrapper around QP solvers in Python",
    url="https://github.com/stephane-caron/qpsolvers",
    author="Stéphane Caron",
    author_email="stephane.caron@normalesup.org",
    license="GPL",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU Lesser General Public License (LGPL)',
        'Programming Language :: Python :: 2.7'],
    packages=['qpsolvers']
)
