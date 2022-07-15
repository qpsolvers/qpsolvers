#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
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

"""
Test all available QP solvers on random quadratic programs.
"""

import sys

try:
    from IPython import get_ipython
except ImportError:
    print("This example requires IPython, try installing ipython3")
    sys.exit(-1)

from numpy import dot, linspace, ones, random
from os.path import basename
from scipy.linalg import toeplitz
from timeit import timeit

from qpsolvers import available_solvers, solve_qp


nb_iter = 10
sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]


def solve_random_qp(n, solver):
    M, b = random.random((n, n)), random.random(n)
    P, q = dot(M.T, M), dot(b, M)
    G = toeplitz(
        [1.0, 0.0, 0.0] + [0.0] * (n - 3), [1.0, 2.0, 3.0] + [0.0] * (n - 3)
    )
    h = ones(n)
    return solve_qp(P, q, G, h, solver=solver)


def plot_results(perfs):
    try:
        from pylab import clf, get_cmap, grid, ion, legend, plot
        from pylab import xlabel, xscale, ylabel, yscale
    except ImportError:
        print("Cannot plot results, try installing python3-matplotlib")
        print("Results are stored in the global `perfs` dictionary")
        return
    cmap = get_cmap("tab10")
    colors = cmap(linspace(0, 1, len(available_solvers)))
    solver_color = {
        solver: colors[i] for i, solver in enumerate(available_solvers)
    }
    ion()
    clf()
    for solver in perfs:
        plot(sizes, perfs[solver], lw=2, color=solver_color[solver])
    grid(True)
    legend(list(perfs.keys()), loc="lower right")
    xscale("log")
    yscale("log")
    xlabel("Problem size $n$")
    ylabel("Time (s)")
    for solver in perfs:
        plot(sizes, perfs[solver], marker="o", color=solver_color[solver])


if __name__ == "__main__":
    if get_ipython() is None:
        print(
            "Run the benchmark with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()
    perfs = {}
    print("\nTesting all QP solvers on random quadratic programs...\n")
    for solver in available_solvers:
        try:
            perfs[solver] = []
            for size in sizes:
                print(f"Running {solver} on problem size {size}...")
                cum_time = timeit(
                    stmt=f"solve_random_qp({size}, '{solver}')",
                    setup="from __main__ import solve_random_qp",
                    number=nb_iter,
                )
                perfs[solver].append(cum_time / nb_iter)
        except Exception as e:
            print(f"Warning: {str(e)}")
            if solver in perfs:
                del perfs[solver]
    plot_results(perfs)
