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
Stochastic NLS by the EULER method, soliton with noise.

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

Versions
--------
Created 14 feb 2020
@author: claudio
@version: 6 february 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS1D as NLS
# %% number of shots
nshots = 10
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 0.01
input['xmin'] = -10.0
input['xmax'] = 10.0
input['nx'] = 256
input['nplot'] = 200
input['nz'] = 100
input['cxx'] = 0.0
input['chi'] = 0.0
input['chinoise'] = 1.0  # chi in noise expression (supersede chi)
input['n0'] = 100  # number of photons
input['noise'] = True  # coefficient to switch noise, if false no noise
input['verbose_level'] = 1
input['plot_level'] = 1
input['step'] = NLS.MILSTEIN_step
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
trajectories = np.zeros((nsteps, nshots), dtype=np.double)
peaks = np.zeros((nsteps, nshots), dtype=complex)

# %% evolve
for ishot in range(nshots):
    # print
    print("Processing shot " + repr(ishot+1) + " of " + repr(nshots))
    # evolve the shots
    out = NLS.evolve_SDE_NLS(input)
    # extract data from sims
    zplot = out['zplot']  # this is not needed
    psi2D = out['psi2D']  # this is not neede
    powers = out['powers']
    mean_xs = out['mean_xs']
    mean_square_xs = out['mean_square_xs']
    # store data
    trajectories[:, ishot] = mean_xs
    peaks[:, ishot] = psi2D[int(input['nx']/2), :]
# %% evaluate mean trajectory
mean_abs2_peak = np.real(np.mean(np.abs(peaks)**2, axis=1))
mean_real_peak = np.real(np.mean(np.cos(peaks), axis=1))
mean_trajectory = np.mean(trajectories, axis=1)
mean_square_trajectory = np.mean(trajectories**2, axis=1)
dXsim = np.sqrt(mean_square_trajectory)
# %% theory
sigmaxx = ((np.pi**2-6)/18.0)*input['chi']/(input['n0']*np.sqrt(beta))
dXth = np.sqrt(sigmaxx*zplot)

# %% peak
figpeak = plt.figure()
plt.plot(zplot, mean_abs2_peak)
plt.ylabel('peak intensity')
plt.xlabel('z')
# %% peak
figcos = plt.figure()
plt.plot(zplot, mean_real_peak)
plt.ylabel('cos intensity')
plt.xlabel('z')
# %% plot last run
fig2D = plt.figure()
plt.pcolormesh(zplot, x, np.abs(psi2D))
plt.ylabel('x')
plt.xlabel('z')
# %% plot trajectories
fig1 = plt.figure()
plt.plot(zplot, trajectories)
plt.plot(zplot, mean_trajectory, linewidth=4, color='k')
plt.ylabel('x')
plt.xlabel('z')
# %% plot mean square x
fig2 = plt.figure()
plt.plot(zplot, np.sqrt(trajectories**2))
plt.plot(zplot, dXsim, linewidth=4, color='k')
# plt.plot(zplot, dXth, linewidth=4, color='r')
plt.ylabel('x')
plt.xlabel('z')

# %% plot mean square x
fig3 = plt.figure()
plt.plot(zplot, dXsim/(dXth+1e-12), linewidth=4, color='k')
plt.ylabel('x')
plt.xlabel('z')
plt.show()

# %% save data
nomefile = './data/MILSTEIN_n0_'+repr(input['n0']) + \
    '_nshots_' + repr(nshots) + \
    '_zmax_' + repr(input['zmax'])
np.savez(nomefile, trajectories, input)
