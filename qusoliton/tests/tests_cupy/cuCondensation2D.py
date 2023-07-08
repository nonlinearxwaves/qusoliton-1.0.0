# -*- coding: utf-8 -*-
"""
Stochastic NLS by the EULER method, soliton with noise, CUDA CUPY version

Created on Thu Dec 26 18:22:38 2019

Solve the stochastic NLS by 

Simulate the 2D evaporative cooling

The equation is written as 

! TODO improve documentation here, fix equations

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
@version: 26 april 2020
"""

import utils
utils.set_project_path()

import time
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

import cupySDENLS.cuSDENLS2D as NLS

#
# %% number of shots
nshots = 2
# %% save data
save_data = False
# %% parameters for evolution (as a dictionary)
input = dict()
input['zmax'] = 10
input['xmin'] = -20.0
input['xmax'] = 20.0
input['ymin'] = -20.0
input['ymax'] = 20.0
input['nx'] = 32
input['ny'] = 32
input['nplot'] = 30
input['nz'] = 1000
input['cxx'] = 1.0
input['cyy'] = 1.0
input['chi'] = -1
input['n0'] = 1000  # number of photons
input['noise'] = True  # coefficient to switch noise, if false no noise
input['verbose_level'] = 1
input['plot_level'] = 1
input['step'] = NLS.MILSTEIN_time_potential_step
# %% coordinates
x, y, _, _ = NLS.coordinates(input)
X, Y = np.meshgrid(x, y)
# %% initial condition
w0x = 1.0  # waist in the x direction
w0y = 1.0  # waist in the y direction
a0 = 1
psi0 = a0*np.exp(-np.square(X/w0x)-np.square(Y/w0y))
# %% add spectral noise to initial condition
amp_noise = 1  # amplitude of spectral noise in [0,1], 1 correspond to pi
psi0 = np.fft.ifft2(
    np.fft.fft2(psi0)*np.exp(
        amp_noise*np.pi*1j*np.random.normal(size=psi0.shape)))
# %% store initial condition
phi0 = np.copy(np.conj(psi0))
input['psi0'] = psi0
input['phi0'] = phi0
# %% allocate the potential
Vmax = 3  # amplitude the maximum potential
Lx = input['xmax']-input['xmin']
Ly = input['ymax']-input['ymin']
potential = np.zeros_like(X)
potential = Vmax*(np.square(np.sin(np.pi*X/Lx))+np.square(np.sin(np.pi*Y/Ly)))
# %% plot potential
figpotential2D = plt.figure()
axp = plt.axes(projection='3d')
axp.plot_surface(X, Y, potential)
axp.set_xlabel("X")
axp.set_ylabel("Y")
axp.set_title("potential")
# %% allocate the absorber
Gmax = 1  # amplitude the maximum potential
absorber = np.zeros_like(X)
absorber = Gmax*(np.power(np.sin(np.pi*X/Lx), 50) +
                 np.power(np.sin(np.pi*Y/Ly), 50))
# %% plot absorber
figabsorber2D = plt.figure()
axa = plt.axes(projection='3d')
axa.plot_surface(X, Y, absorber)
axa.set_xlabel("X")
axa.set_ylabel("Y")
axa.set_title("absorber")
# %% store potential and absorber
input['potential'] = potential
input['absorber'] = absorber
# %% allocate space for the shots
nsteps = input['nplot']+1  # stored steps
nx = input['nx']
ny = input['ny']
mean_intens = cp.zeros((ny, nx, nsteps))
mean_kintens = cp.zeros((ny, nx, nsteps))
# %% evolve
for ishot in range(nshots):
    # print
    print("Processing shot " + repr(ishot+1) + " of " + repr(nshots))
    print("--------------------------------")
    # evolve the shots
    out = NLS.evolve_SDE_NLS(input)
    # extract data from sims
    zplot = out['zplot']  # this is not needed
    powers = out['powers']
    mean_xs = out['mean_xs']
    mean_square_xs = out['mean_square_xs']
    # store dat
    mean_intens = mean_intens + out['intens']
    mean_kintens = mean_kintens + out['kintens']


# scale the mean quantities with the number of shots
inv_nshots = 1.0/nshots
mean_intens = mean_intens*inv_nshots
mean_kintens = mean_kintens*inv_nshots

# %% plot last energy density and spectral density
iplot = input['nplot']  # last point
tmpplot1 = cp.asnumpy(mean_intens[:, :, iplot])
tmpplot2 = np.fft.fftshift(cp.asnumpy(mean_kintens[:, :, iplot]))
fig = plt.figure(figsize=plt.figaspect(0.5))
axa = fig.add_subplot(1, 2, 1, projection='3d')
axa.set_xlabel('X')
axa.set_ylabel('Y')
axa.set_title("mean energy density")
axb = fig.add_subplot(1, 2, 2, projection='3d')
axb.set_title("mean spectral density")
axb.set_xlabel('Kx')
axb.set_ylabel('Ky')
surf_intens = axa.plot_surface(out['X'], out['Y'], tmpplot1)
surf_kintens = axb.plot_surface(out['KX'], out['KY'], tmpplot2)
plt.show()


# %% save data
if save_data:
    nomefile = 'cuCondensate2D_n0_'+repr(input['n0']) + \
        '_nshots_' + repr(nshots) + \
        '_zmax_' + repr(input['zmax'])
    np.savez(nomefile, mean_intens, mean_kintes, input)
