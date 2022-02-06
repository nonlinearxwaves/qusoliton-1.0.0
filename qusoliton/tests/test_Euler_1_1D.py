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
Solve the stochastic NLS by the EULER method, test diffraction, no noise.

Created on Thu Dec 26 18:22:38 2019.

The equation is written as

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx phi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 1, test the diffraction do a gaussian beam with parameters
chi = 0
cxx = 1
input beam is exp( -x**2  )
the expected trend for the waist w0
w = w0 np.sqrt[1+16 cxx**2 z**2 w0**-4]
and the std = w/2.0

@author: claudio
@version: 14 march 2020
"""

# TODO: test the diffraction with a gaussian beam
import time
from qusoliton.sdenls import SDENLS1D as NLS
import matplotlib.pyplot as plt
import numpy as np
#from os.path import dirname, realpath, sep, pardir
#import sys
#sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "SDENLS")

#
# %% timing
startt = time.time()
# %% parameters (as a dictionary)
input = dict()
input['zmax'] = np.pi
input['xmin'] = -30.0
input['xmax'] = 30.0
input['nx'] = 256
input['nplot'] = 40
input['nz'] = 1000
input['cxx'] = 1.5
input['chi'] = 0.0
input['n0'] = 1000  # number of photons
input['noise'] = False  # coefficient to switch noise, if false no noise
input['plot_level'] = 2
input['verbose_level'] = 2
input['step'] = NLS.MILSTEIN_step
# %% coordinates
x, _ = NLS.coordinates(input)
# %% initial condition
w0 = 2.0  # initial waist
psi0 = np.exp(-np.square(x/w0))
phi0 = psi0
input['psi0'] = psi0
input['phi0'] = phi0
# %% evolve
out = NLS.evolve_SDE_NLS(input)
# %% extract data from sims
zplot = out['zplot']
psi2D = out['psi2D']
powers = out['powers']
mean_xs = out['mean_xs']
mean_square_xs = out['mean_square_xs']
# %%
fig2D = plt.figure()
plt.pcolormesh(zplot, x, np.abs(psi2D))
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
plt.xlabel('z')
# %% plot the std and compare with exact solution
cxx = input['cxx']
std_ex = (w0/2.0)*np.sqrt(1.0+16.0*cxx*cxx*(zplot/w0**2)**2)
plt.figure()
plt.plot(zplot, np.sqrt(mean_square_xs))
plt.plot(zplot, std_ex, 'rx')
plt.ylabel('sqrt(<x**2>)')
plt.xlabel('z')
plt.show()
# %% timing
endt = time.time()
if input['verbose_level'] > 0:
    print('Total time (s) '+repr(endt-startt))
