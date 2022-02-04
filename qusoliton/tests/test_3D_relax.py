# This file is part of QuSoliton: Quantum Soliton Toolbox.
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
# test relaxation in 3D
""" Created 5 dic 2021
"""


import numpy as np
from PIL import Image
import time
from qusoliton.sdenls import SDENLS3D as NLS
import sys
from termcolor import colored


#from importlib import reload
# reload(NLS)
# %% start timing
startt = time.time()

# %% plot_level
plot_level = 2

# save data flag
save_relax_data = False
filename_relax = './data/relaxdata'

# %% define parameters
npoint = 32  # common point in any direction
NLS.xmin = -25
NLS.xmax = 25
NLS.nx = npoint
NLS.ymin = -25
NLS.ymax = 25
NLS.ny = npoint
NLS.tmin = -25
NLS.tmax = 25
NLS.nt = npoint
NLS.chi = 1.0
NLS.nplot = 10
NLS.nplot = 10
NLS.nz = 100
NLS.zmax = 1.0
NLS.verbose_level = 2
NLS.plot_level = 0
NLS.iter_implicit = 0
NLS.sigma_local = 1
NLS.sigma_nonlocal = 1
NLS.alphanoise = 0
NLS.nphoton = 100
# define main algol
NLS.step = NLS.DRUMMOND_step
NLS.iter_implicit = 3
# parameter for relax
NLS.relax_eig_tol = 1.0e-6
NLS.relax_tol = 1.0e-13
NLS.relax_alpha = 0.1
NLS.relax_iter_linear = 1000
NLS.relax_niter = 1000
# init module
NLS.init()

# initial condition
w0x = 5.66  # initial waist
w0y = 5.66  # initial waist
w0t = 5.66  # initial waist
a0 = 1.0
NLS.psi0 = a0*np.exp(-np.square(NLS.X/w0x) -
                     np.square(NLS.Y/w0y)-np.square(NLS.T/w0t))
NLS.phi0 = np.copy(NLS.psi0)


# %% plot initial condition
if plot_level > 0:
    figure_name = './img/initialguess.png'
    Image.open(NLS.plot_panels(
        NLS.psi0, figure_name, 'initial condition')).show()

# %% relax
out = NLS.relax()
# out = NLS.relax_tests()
# %%
if plot_level > 0:
    Image.open(
        NLS.plot_panels(np.abs(NLS.psi0),
                        './img/relaxprofile.png',
                        'final state after relaxation'
                        )).show()

# save the variables with the relaxation profile
if save_relax_data:
    NLS.saveall(filename_relax)
# evolve the state
outdata = NLS.evolve_SDE_NLS(input)

if plot_level > 0:
    Image.open(
        NLS.plot_panels(np.abs(NLS.psi),
                        './img/finalepsi.png',
                        'final |psi| after evolution'
                        )).show()


# reload the data and evolve to test
if save_relax_data:
    NLS.phi0 = None
    NLS.psi0 = None
    print(colored('Reloading and evolving again', 'blue'))
    NLS.loadall(filename_relax+'.npz')
    # evolve the state after turning on noise
    NLS.alphanoise = 1
    NLS.nphoton = 1000
    NLS.init()  # reinit module after changing parameters
    outdata = NLS.evolve_SDE_NLS(input)

    if plot_level > 0:
        Image.open(
            NLS.plot_panels(np.abs(NLS.psi),
                            './img/finalepsi.png',
                            'final |psi| after evolution'
                            )).show()
        Image.open(
            NLS.plot_observables(
                './img/observables.png',
                'observables'
            )).show()
