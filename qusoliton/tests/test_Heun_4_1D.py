# This file is part of QuSoliton: Quantum Soliton toolbox.
#
# Copyright (c) 2022 and later, Claudio Conti.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the QuSoliton : Quantum Soliton toolbox nor the names of
#  its contributors may be used to endorse or promote products derived from this
#  software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:22:38 2019

Solve the stochastic NLS by the HEUN method

The equation is written as

  psi_t =  +1j c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
  phi_t =  -1j c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 3, test the stochastic fluctuactions of fundamental soliton
chi = 1
cxx = 1
input beam is sech( x )
which is propagation invariant in absence of noise

nshots evolutions are done and the trajectories are stored and compared

make various simulations for different values of n0

Load saved data
a=np.load('HEUN1D_n0s_20_nshots_2_zmax_1.npz', allow_pickle=True)

@author: claudio
@created: 27 january 2020
@version: 6 february 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS1D as NLS
# %% number of shots
nshots = 10
# %% vector of n0 values
num_n0 = 10  # number of points of n0
n0s = np.linspace(100, 1000, num=num_n0)
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 0.1
input['xmin'] = -20.0
input['xmax'] = 20.0
input['nx'] = 256
input['nplot'] = 100
input['nz'] = 10
input['cxx'] = 1.0
input['chi'] = 1.0
input['n0'] = 10000  # number of photons
input['noise'] = True  # coefficient to switch noise, if false no noise
input['verbose_level'] = 1
input['plot_level'] = 0
# %% coordinates
x, _ = NLS.coordinates(input)
# %% initial condition
psi0 = np.reciprocal(np.cosh(x), dtype=np.double)
phi0 = psi0
input['psi0'] = psi0
input['phi0'] = phi0
# %% allocate space for the shots
nsteps = input['nplot']+1  # stored steps
trajectories = np.zeros((nsteps, nshots, num_n0), dtype=np.double)
mean_trajectories = np.zeros((nsteps, num_n0), dtype=np.double)
# %% evolve
for in0 in range(num_n0):
    input['n0'] = n0s[in0]  # set current n0
    for ishot in range(nshots):
        # print
        print("Processing shot " + repr(ishot+1) + " of " + repr(nshots)
              + " for n0 " + repr(n0s[in0]) + ", " + repr(in0+1) + " of " +
              repr(num_n0))
        # evolve the shots
        out = NLS.evolve_SDE_NLS_Heun(input)
        # extract data from sims
        zplot = out['zplot']  # this is not needed
        powers = out['powers']
        mean_xs = out['mean_xs']
        mean_square_xs = out['mean_square_xs']
        # store dat
        trajectories[:, ishot, in0] = mean_xs
    mean_trajectories[:, in0] = np.mean(trajectories**2, axis=1)[:, in0]

# %% plot trajectories
fig1 = plt.figure()
# plt.plot(zplot, trajectories)
plt.plot(zplot, mean_trajectories, linewidth=4)
plt.ylabel('<x2>')
plt.xlabel('z')

# %% final values vs n0, scale with t and plot theory
fig2 = plt.figure()
plt.plot(n0s, np.transpose(mean_trajectories), linewidth=1)
plt.ylabel('<x2>')
plt.xlabel('n_0')
plt.show()

# %% save data
nomefile = './data/HEUN1D_n0s_'+repr(num_n0) + \
    '_nshots_' + repr(nshots) + \
    '_zmax_' + repr(input['zmax'])
np.savez(nomefile, trajs=trajectories,
         meant=mean_trajectories,
         n0s=n0s,
         idta=input)
