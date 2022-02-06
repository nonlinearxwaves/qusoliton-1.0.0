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
Solve the stochastic NLS by the EULER method, test collapse with noise

Created on Thu Dec 26 18:22:38 2019.

The equation is written as

+1j psi_t + c_xx psi_xx + c_yy psi_yy + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
-1j phi_t + c_xx phi_xx + c_yy phi_yy- 1j 2.0 chi psi phi^2 + nphi phi rand_phi

TEST 1, test the collapse of a gaussian beam with parameters
chi = 1
cxx = 1
cyy = 1
input beam is exp( -x**2 --x**2  )
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
# %% shots
nshots = 1
# %% parameters (as a dictionary)
input = dict()
input['zmax'] = 0.4
input['xmin'] = -5.0
input['xmax'] = 5.0
input['ymin'] = -5.0
input['ymax'] = 5.0
input['nx'] = 64
input['ny'] = 128
input['nplot'] = 40
input['nz'] = 200
input['cxx'] = 1.0
input['cyy'] = 1.0
input['chi'] = 1.0
input['n0'] = 1000  # number of photons
input['noise'] = False  # coefficient to switch noise, if false no noise
input['plot_level'] = 2
input['verbose_level'] = 1
input['step'] = NLS.MILSTEIN_step
# %% coordinates
x, y, _, _ = NLS.coordinates(input)
X, Y = np.meshgrid(x, y)
# %% initial condition
w0x = 1.0  # waist in the x direction
w0y = 1.0  # waist in the y direction
a0 = 2.0  # amplitude ( must be greater than 2 to have collapse )
psi0 = a0*np.exp(-np.square(X/w0x)-np.square(Y/w0y))
phi0 = psi0
input['psi0'] = psi0
input['phi0'] = phi0

# %% allocate space for the shots
nsteps = input['nplot']+1  # stored steps
waists = np.zeros((nsteps, nshots), dtype=np.float64)
peaks = np.zeros((nsteps, nshots), dtype=np.float64)
# %% evolve any shots
for ish in range(nshots):
    print('Processing shot ' + repr(ish+1) + ' of ' + repr(nshots))
    out = NLS.evolve_SDE_NLS(input)
    # %% extract data from sims
    peaks[:, ish] = out['peaks']
    waists[:, ish] = out['mean_square_rs']
# %% extract zplot from sims
zplot = out['zplot']
# %% plot observables
plt.figure()
plt.plot(zplot, peaks)
plt.ylabel('peak')
plt.xlabel('z')
plt.figure()
plt.plot(zplot, np.sqrt(waists))
plt.ylabel('waist')
plt.xlabel('z')
plt.show()
# %% timing
endt = time.time()
if input['verbose_level'] > 0:
    print('Total time (s) '+repr(endt-startt))
