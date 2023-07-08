# -*- coding: utf-8 -*-
"""
Stochastic NLS by the EULER method, soliton with noise, CUDA CUPY version

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
Created 9 april 2020
@author: claudio
@version: 2 gennaio 2022
"""


# TODO: test the diffraction with a gaussian beam
import sys
sys.path.append("../")

# 
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cupySDENLS.cuSDENLS1D as NLS
import time
# %% number of shots
nshots = 1
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 0.01
input['xmin'] = -20.0
input['xmax'] = 20.0
input['nx'] = 128
input['nplot'] = 10
input['nz'] = 1000
input['cxx'] = 1.0
input['chi'] = 1.0
input['n0'] = 10000  # number of photons
input['noise'] = True  # coefficient to switch noise, if false no noise
input['verbose_level'] = 1
input['plot_level'] = 1
input['step'] = NLS.EULER_step
# %% coordinates
x = NLS.coordinates(input)
# %% initial condition for the fundamental soliton
beta = 1  # soliton parameter
psi0 = np.sqrt(beta)*np.reciprocal(np.cosh(x*np.sqrt(beta)), dtype=np.complex)
phi0 = psi0
input['psi0'] = psi0
input['phi0'] = phi0
# %% allocate space for the shots
nsteps = input['nplot']+1  # stored steps
trajectories = np.zeros((nsteps, nshots), dtype=np.complex)
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
    # store dat
    trajectories[:, ishot] = mean_xs

# %% evaluate mean trajectory
R_trajectories = np.real(trajectories)
mean_trajectory = np.mean(R_trajectories, axis=1)

mean_square_trajectory = np.mean(R_trajectories**2, axis=1)
dXsim = np.sqrt(mean_square_trajectory)
# %% theory
sigmaxx = ((np.pi**2-6)/18.0)*input['chi']/(input['n0']*np.sqrt(beta))
dXth = np.sqrt(sigmaxx*zplot)

# %% plot last run
fig2D = plt.figure()
plt.pcolormesh(zplot, x, np.abs(psi2D))
plt.ylabel('x')
plt.xlabel('z')

# %% plot trajectories
fig1 = plt.figure()
plt.plot(zplot, R_trajectories)
plt.plot(zplot, mean_trajectory, linewidth=4, color='k')
plt.ylabel('x')
plt.xlabel('z')
# %% plot mean square x
fig2 = plt.figure()
plt.plot(zplot, np.real(np.sqrt(trajectories**2)))
plt.plot(zplot, dXsim, linewidth=4, color='k')
plt.plot(zplot, dXth, linewidth=4, color='r')
plt.ylabel('x')
plt.xlabel('z')


# %% save data
nomefile = 'MILSTEIN_n0_'+repr(input['n0']) + \
        '_nshots_' + repr(nshots) + \
        '_zmax_' + repr(input['zmax'])
np.savez(nomefile, trajectories, input)
