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
Solve the stochastic NLS by the EULER method, test soliton, no noise.

Created on Thu Dec 26 18:22:38 2019

Solve the stochastic NLS by the HEUN method

The equation is written as

  psi_t =  +1j c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
  phi_t =  -1j c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 2, test the deterministic fundamental soliton
chi = 1
cxx = 1
input beam is sech( x )
which is propagation invariant in absence of noise


@author: claudio
@created: 06 january 2020 at Snowbird PQE50
@version: 17 february 2020
"""


import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS1D as NLS
# %% parameters (as a dictionary)
input = dict()
input['zmax'] = np.pi
input['xmin'] = -20.0
input['xmax'] = 20.0
input['nx'] = 256
input['nplot'] = 30
input['nz'] = 10000
input['cxx'] = 1.0
input['chi'] = 1.0
input['n0'] = 1000  # number of photons
input['noise'] = False  # coefficient to switch noise, if false no noise
input['plot_level'] = 2
input['verbose_level'] = 2
input['step'] = NLS.EULER_step
# %% coordinates
x, _ = NLS.coordinates(input)
# %% initial condition
psi0 = np.reciprocal(np.cosh(x), dtype=np.double)
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
zR = 0.5
std_ex = np.sqrt(2.0)*np.sqrt(1.0+(zplot/zR)**2)
plt.figure()
plt.plot(zplot, np.sqrt(mean_square_xs))
plt.plot(zplot, std_ex, 'r')
plt.ylabel('sqrt(<x**2>)')
plt.xlabel('z')
plt.show()
