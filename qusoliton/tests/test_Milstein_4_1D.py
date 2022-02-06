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

Solve the stochastic NLS by the Milstein method

The equation is written as

  psi_t =  +1j c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
  phi_t =  -1j c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 4, test the stochastic fluctuactions of fundamental soliton
chi = 1
cxx = 1
input beam is sech( x )
which is propagation invariant in absence of noise

nshots evolutions are done and the trajectories are stored and compared

make various simulations for different values of n0

To load the .npz data
a=np.load(file)
a.files

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
num_n0 = 30  # number of points of n0
n0s = np.linspace(1000, 10000, num=num_n0)
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 0.01
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
input['step'] = NLS.MILSTEIN_step
# %% flag for saving data
save_data = False
# %% coordinates
x, _ = NLS.coordinates(input)
# %% initial condition for the fundamental soliton
beta = 1  # soliton parameter
psi0 = np.sqrt(beta)*np.reciprocal(np.cosh(x*np.sqrt(beta)), dtype=np.double)
phi0 = psi0
input['psi0'] = psi0
input['phi0'] = phi0
# %% allocate space for the shots
nsteps = input['nplot']+1  # stored steps
trajectories = np.zeros((nsteps, nshots, num_n0), dtype=complex)
trajectories2 = np.zeros((nsteps, nshots, num_n0), dtype=complex)
mean_trajectories = np.zeros((nsteps, num_n0), dtype=complex)
# %% evolve
for in0 in range(num_n0):
    input['n0'] = n0s[in0]  # set current n0
    for ishot in range(nshots):
        # print
        print("Processing shot " + repr(ishot+1) + " of " + repr(nshots)
              + " for n0 " + repr(n0s[in0]) + ", " + repr(in0+1) + " of " +
              repr(num_n0))
        # evolve the shots
        out = NLS.evolve_SDE_NLS(input)
        # extract data from sims
        zplot = out['zplot']  # this is not needed
        powers = out['powers']
        mean_xs = out['mean_xs']
        mean_square_xs = out['mean_square_xs']
        # store dat
        trajectories[:, ishot, in0] = mean_xs
        trajectories2[:, ishot, in0] = mean_square_xs
        mean_trajectories[:, in0] = np.sqrt(
            np.mean(trajectories**2, axis=1)[:, in0])

# %% plot trajectories
fig0 = plt.figure()
plt.plot(zplot, np.real(trajectories[:, :, -1]))
plt.ylabel('x')
plt.xlabel('z')

# %% plot mean trajectories with theory
# superimpose theory
mean_tr_th = np.zeros((nsteps, num_n0), dtype=np.double)
for in0 in range(num_n0):
    sigmaxx = (1/18)*(np.pi**2-6.0)/(n0s[in0]*np.sqrt(beta))
    mean_tr_th[:, in0] = np.sqrt(sigmaxx*zplot)
# %% plot
fig1 = plt.figure()
plt.plot(zplot, np.real(mean_trajectories), linewidth=4)
plt.plot(zplot, mean_tr_th, 'x', linewidth=1)
plt.ylabel('\sqrt(<X2>)')
plt.xlabel('z')

# %% final values vs n0, scale with t and plot theory
fig2 = plt.figure()
plt.plot(n0s, np.real(mean_trajectories[-1, :]), linewidth=1)
plt.plot(n0s, mean_tr_th[-1, :], 'x', linewidth=1)
plt.ylabel('\sqrt<X2>')
plt.xlabel('n_0')
plt.show()

# %% save data
if save_data:
    nomefile = './data/MILSTEIN1D_n0s_'+repr(num_n0) + \
        '_nshots_' + repr(nshots) + \
        '_zmax_' + repr(input['zmax']) + \
        '_beta_' + repr(beta)
    np.savez(nomefile, trajectories=trajectories,
             mean_trajs=mean_trajectories,
             n0s=n0s,
             idata=input)
    print(" Saved "+repr(nomefile))
