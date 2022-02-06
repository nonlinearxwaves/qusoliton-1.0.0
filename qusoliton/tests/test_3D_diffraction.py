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
# test diffraction in 3D
""" Created 20 nov 2021
"""


# set project path
import numpy as np
from qusoliton.sdenls import SDENLS3D as NLS
import time
from PIL import Image


# %% start timing
startt = time.time()

# %% plot_level
plot_level = 1

# %% define parameters as dictionary

npoint = 16  # common point in any direction
NLS.zmax = 1.0
NLS.xmin = -10
NLS.xmax = 10
NLS.nx = npoint
NLS.ymin = -10
NLS.ymax = 10
NLS.ny = npoint
NLS.tmin = -10
NLS.tmax = 10
NLS.nt = npoint
NLS.nplot = 10
NLS.nz = 100
NLS.chi = 0
NLS.nphoton = 1000
NLS.alphanoise = 0
NLS.verbose_level = 2
NLS.plot_level = 0
NLS.iter_implicit = 0
NLS.sigma_local = 1
NLS.sigma_nonlocal = 1
# define main algol
NLS.step = NLS.DRUMMOND_step


# init module
NLS.init()

# initial condition
w0x = 2.0  # initial waist
w0y = 2.0  # initial waist
w0t = 2.0  # initial waist
psi0 = np.exp(-np.square(NLS.X/w0x)-np.square(NLS.Y/w0y)-np.square(NLS.T/w0t))
phi0 = np.copy(psi0)
NLS.psi0 = psi0
NLS.phi0 = phi0
# plot initial conditions

# %% 2D matrix 2 plot
#matpsi0xy = psi0[:, :, NLS.nt//2]
#matpsi0xt = psi0[:, NLS.ny//2, :]
#matpsi0yt = psi0[NLS.nx//2, :, :]


# %% plot initial condition
if plot_level > 0:
    figure_name = './img/initialcondition.png'
    Image.open(NLS.plot_panels(psi0, figure_name, 'initial condition')).show()

# %% evolve
out = NLS.evolve_SDE_NLS(input)

# %%
Image.open(
    NLS.plot_panels(np.abs(NLS.psi), './img/currentplot.png',
                    'current status'
                    )).show()
Image.open(
    NLS.plot_observables(
        './img/observables.png',
        'observables'
    )).show()
