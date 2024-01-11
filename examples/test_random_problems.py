#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test all available QP solvers on random quadratic programs."""

import sys

try:
    from IPython import get_ipython
except ImportError:
    print("This example requires IPython, try installing ipython3")
    sys.exit(-1)

from os.path import basename
from timeit import timeit

from numpy import dot, linspace, ones, random
from qpsolvers import available_solvers, solve_qp
from scipy.linalg import toeplitz

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
        from pylab import (
            clf,
            get_cmap,
            grid,
            ion,
            legend,
            plot,
            xlabel,
            xscale,
            ylabel,
            yscale,
        )
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
            "This example should be run with IPython:\n\n"
            f"\tipython -i {basename(__file__)}\n"
        )
        exit()
    perfs = {}

    benchmark = "https://github.com/qpsolvers/qpbenchmark"
    print("\nTesting all solvers on a given set of random QPs")
    print(f"For a proper benchmark, check out {benchmark}")

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
