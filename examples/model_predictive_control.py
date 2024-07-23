#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Test the "quadprog" QP solver on a model predictive control problem.

The problem is to balance a humanoid robot walking on a flat horizontal floor.
See the following post for context:

    https://scaron.info/robot-locomotion/prototyping-a-walking-pattern-generator.html
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

import qpsolvers
from qpsolvers import solve_qp

gravity = 9.81  # [m] / [s]^2


@dataclass
class HumanoidSteppingProblem:

    com_height: float = 0.8
    dsp_duration: float = 0.1
    end_pos: float = 0.3
    foot_length: float = 0.1
    horizon_duration: float = 2.5
    nb_timesteps: int = 16
    ssp_duration: float = 0.7
    start_pos: float = 0.0


class LinearModelPredictiveControl:

    """
    Linear model predictive control for a system with linear dynamics and
    linear constraints. This class is fully documented at:

        https://scaron.info/doc/pymanoid/walking-pattern-generation.html#pymanoid.mpc.LinearPredictiveControl
    """

    def __init__(
        self,
        A,
        B,
        C,
        D,
        e,
        x_init,
        x_goal,
        nb_timesteps: int,
        wxt: Optional[float],
        wxc: Optional[float],
        wu: float,
    ):
        assert C is not None or D is not None, "use LQR for unconstrained case"
        assert (
            wu > 0.0
        ), "non-negative control weight needed for regularization"
        assert wxt is not None or wxc is not None, "set either wxt or wxc"
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.G = None
        self.P = None
        self.U = None
        self.U_dim = u_dim * nb_timesteps
        self.e = e
        self.h = None
        self.nb_timesteps = nb_timesteps
        self.q = None
        self.u_dim = u_dim
        self.wu = wu
        self.wxc = wxc
        self.wxt = wxt
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init
        #
        self.build()

    def build(self):
        phi = np.eye(self.x_dim)
        psi = np.zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in range(self.nb_timesteps):
            # Loop invariant: x == psi * U + phi * x_init
            if self.wxc is not None:
                phi_list.append(phi)
                psi_list.append(psi)
            C = self.C[k] if type(self.C) is list else self.C
            D = self.D[k] if type(self.D) is list else self.D
            e = self.e[k] if type(self.e) is list else self.e
            G = np.zeros((e.shape[0], self.U_dim))
            h = e if C is None else e - np.dot(C.dot(phi), self.x_init)
            if D is not None:
                # we rely on G == 0 to avoid a slower +=
                G[:, k * self.u_dim : (k + 1) * self.u_dim] = D
            if C is not None:
                G += C.dot(psi)
            if k == 0 and D is None:  # corner case, input has no effect
                assert np.all(h >= 0.0)
            else:  # regular case
                G_list.append(G)
                h_list.append(h)
            phi = self.A.dot(phi)
            psi = self.A.dot(psi)
            psi[:, self.u_dim * k : self.u_dim * (k + 1)] = self.B
        P = self.wu * np.eye(self.U_dim)
        q = np.zeros(self.U_dim)
        if self.wxt is not None and self.wxt > 1e-10:
            c = np.dot(phi, self.x_init) - self.x_goal
            P += self.wxt * np.dot(psi.T, psi)
            q += self.wxt * np.dot(c.T, psi)
        if self.wxc is not None and self.wxc > 1e-10:
            Phi = np.vstack(phi_list)
            Psi = np.vstack(psi_list)
            X_goal = np.hstack([self.x_goal] * self.nb_timesteps)
            c = np.dot(Phi, self.x_init) - X_goal
            P += self.wxc * np.dot(Psi.T, Psi)
            q += self.wxc * np.dot(c.T, Psi)
        self.P = P
        self.q = q
        self.G = np.vstack(G_list)
        self.h = np.hstack(h_list)
        self.P_csc = csc_matrix(self.P)
        self.G_csc = csc_matrix(self.G)

    def solve(self, solver: str, sparse: bool = False, **kwargs):
        P = self.P_csc if sparse else self.P
        G = self.G_csc if sparse else self.G
        U = solve_qp(P, self.q, G, self.h, solver=solver, **kwargs)
        self.U = U.reshape((self.nb_timesteps, self.u_dim))

    @property
    def states(self):
        assert self.U is not None, "you need to solve() the MPC problem first"
        X = np.zeros((self.nb_timesteps + 1, self.x_dim))
        X[0] = self.x_init
        for k in range(self.nb_timesteps):
            X[k + 1] = self.A.dot(X[k]) + self.B.dot(self.U[k])
        return X


class HumanoidModelPredictiveControl(LinearModelPredictiveControl):
    def __init__(self, problem: HumanoidSteppingProblem):
        T = problem.horizon_duration / problem.nb_timesteps
        nb_init_dsp_steps = int(round(problem.dsp_duration / T))
        nb_init_ssp_steps = int(round(problem.ssp_duration / T))
        nb_dsp_steps = int(round(problem.dsp_duration / T))
        state_matrix = np.array(
            [[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
        )
        control_matrix = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T])
        control_matrix = control_matrix.reshape((3, 1))
        zmp_from_state = np.array([1.0, 0.0, -problem.com_height / gravity])
        ineq_matrix = np.array([+zmp_from_state, -zmp_from_state])
        cur_max = problem.start_pos + 0.5 * problem.foot_length
        cur_min = problem.start_pos - 0.5 * problem.foot_length
        next_max = problem.end_pos + 0.5 * problem.foot_length
        next_min = problem.end_pos - 0.5 * problem.foot_length
        ineq_vector = [
            np.array([+1000.0, +1000.0])
            if i < nb_init_dsp_steps
            else np.array([+cur_max, -cur_min])
            if i - nb_init_dsp_steps <= nb_init_ssp_steps
            else np.array([+1000.0, +1000.0])
            if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps
            else np.array([+next_max, -next_min])
            for i in range(problem.nb_timesteps)
        ]
        super().__init__(
            state_matrix,
            control_matrix,
            ineq_matrix,
            None,
            ineq_vector,
            x_init=np.array([problem.start_pos, 0.0, 0.0]),
            x_goal=np.array([problem.end_pos, 0.0, 0.0]),
            nb_timesteps=problem.nb_timesteps,
            wxt=1.0,
            wxc=None,
            wu=1e-3,
        )


def plot_mpc_solution(problem, mpc):
    t = np.linspace(0.0, problem.horizon_duration, problem.nb_timesteps + 1)
    X = mpc.states
    zmp_from_state = np.array([1.0, 0.0, -problem.com_height / gravity])
    zmp = X.dot(zmp_from_state)
    pos = X[:, 0]
    zmp_min = [x[0] if abs(x[0]) < 10 else None for x in mpc.e]
    zmp_max = [-x[1] if abs(x[1]) < 10 else None for x in mpc.e]
    zmp_min.append(zmp_min[-1])
    zmp_max.append(zmp_max[-1])
    plt.ion()
    plt.clf()
    plt.plot(t, pos)
    plt.plot(t, zmp, "r-")
    plt.plot(t, zmp_min, "g:")
    plt.plot(t, zmp_max, "b:")
    plt.grid(True)
    plt.show(block=True)


if __name__ == "__main__":
    problem = HumanoidSteppingProblem()
    mpc = HumanoidModelPredictiveControl(problem)
    mpc.solve(solver=random.choice(qpsolvers.available_solvers))
    plot_mpc_solution(problem, mpc)
