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

@author: claudio
@version: 07 january 2020 at Snowbird PQE50
@version: 06 february 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS1D as NLS
# %% number of shots
nshots = 5
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 1
input['xmin'] = -20.0
input['xmax'] = 20.0
input['nx'] = 512
input['nplot'] = 100
input['nz'] = 100
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
trajectories = np.zeros((nsteps, nshots), dtype=np.double)
# %% evolve
for ishot in range(nshots):
    # print
    print("Processing shot " + repr(ishot+1) + " of " + repr(nshots))
    # evolve the shots
    out = NLS.evolve_SDE_NLS_Heun(input)
    # extract data from sims
    zplot = out['zplot']  # this is not needed
    psi2D = out['psi2D']  # this is not neede
    powers = out['powers']
    mean_xs = out['mean_xs']
    mean_square_xs = out['mean_square_xs']
    # store dat
    trajectories[:, ishot] = mean_xs
# %% evaluate mean trajectory
mean_trajectory = np.mean(trajectories, axis=1)
mean_square_trajectory = np.mean(trajectories**2, axis=1)

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
plt.plot(zplot, trajectories**2)
plt.plot(zplot, mean_square_trajectory, linewidth=4, color='k')
plt.ylabel('x')
plt.xlabel('z')
plt.show()


# %% save data
nomefile = './data/n0_'+repr(input['n0']) + \
    '_nshots_' + repr(nshots) + \
    '_zmax_' + repr(input['zmax'])
np.savez(nomefile, trajectories, input)
