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
Solve the stochastic NLS by the EULER method, test diffraction, no noise, 2D

Created on Thu Dec 26 18:22:38 2019.

The equation is written as

+1j psi_t + c_xx psi_xx + c_yy psi_yy + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
-1j phi_t + c_xx phi_xx + c_yy phi_yy- 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 1, test the diffraction do a gaussian beam with parameters
chi = 0
cxx = 0.75
cyy = 1.5
input beam is exp( -(x/w0x)**2 -(y/w0y)**2  )
the expected trend for the waist
w = np.sqrt[1+16 z**2]   |  # TODO check this formula
and the std = w/2.0

@author: claudio
@version: 6 february 2022
"""

# TODO: test the diffraction with a gaussian beam

import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS2D as NLS
import time
# %% timing
startt = time.time()
# %% parameters (as a dictionary)


input = dict()
input['zmax'] = 0.1
input['xmin'] = -22.0
input['xmax'] = 27.0
input['ymin'] = -18.0
input['ymax'] = 20.0
input['nx'] = 512
input['ny'] = 512
input['nplot'] = 3
input['nz'] = 200
input['cxx'] = 0.75
input['cyy'] = 1.5
input['chi'] = 0.0
input['n0'] = 1000  # number of photons
input['noise'] = False  # coefficient to switch noise, if false no noise
input['plot_level'] = 2
input['verbose_level'] = 2
input['step'] = NLS.EULER_step
# %% coordinates
x, y, _, _ = NLS.coordinates(input)
X, Y = np.meshgrid(x, y)
# %% initial condition
w0x = 2.0  # waist in the x direction
w0y = 1.0  # waist in the y direction
psi0 = np.exp(-np.square(X/w0x)-np.square(Y/w0y))
phi0 = np.copy(psi0)
input['psi0'] = psi0
input['phi0'] = phi0
# %% plot initial condiiton
# %% evolve
out = NLS.evolve_SDE_NLS(input)
# %% extract data from sims
zplot = out['zplot']
# psi2D = out['psi2D']
powers = out['powers']
mean_xs = out['mean_xs']
mean_ys = out['mean_ys']
mean_square_xs = out['mean_square_xs']
mean_square_ys = out['mean_square_ys']
mean_square_rs = out['mean_square_rs']
# %%
fig2D = plt.figure()
# plt.pcolormesh(zplot, x, np.abs(psi2D))
plt.ylabel('x')
plt.xlabel('z')
# %% plot observables
plt.figure()
plt.plot(zplot, powers)
plt.ylabel('power')
plt.xlabel('z')
plt.figure()
plt.plot(zplot, mean_xs)
plt.ylabel('<x>')
plt.figure()
plt.plot(zplot, mean_ys)
plt.ylabel('<y>')
plt.xlabel('z')
plt.figure()
# %% plot the std and compare with exact solution
cxx = input['cxx']
cyy = input['cyy']
# exact solution for unitary waist Gaussian
std_exx = (w0x/2.0)*np.sqrt(1.0+16.0*cxx*cxx*(zplot/w0x**2)**2)
std_eyy = (w0y/2.0)*np.sqrt(1.0+16.0*cyy*cyy*(zplot/w0y**2)**2)
std_e = np.sqrt(std_exx**2+std_eyy**2)
plt.figure()
plt.plot(zplot, np.sqrt(mean_square_xs), 'k')
plt.plot(zplot, np.sqrt(mean_square_ys), 'r')
plt.plot(zplot, np.sqrt(mean_square_rs), 'b')
plt.plot(zplot, std_exx, 'kx')
plt.plot(zplot, std_eyy, 'rx')
plt.plot(zplot, std_e, 'bx')
plt.ylabel('sqrt(<x**2>)')
plt.xlabel('z')
plt.show()
# %% timing
endt = time.time()
if input['verbose_level'] > 0:
    print('Total time (s) '+repr(endt-startt))
