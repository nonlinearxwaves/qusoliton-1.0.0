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
# test relaxation in 3D and evolve with noise
# REMARK data are created by test_3D_relax.py
""" Created 31 dic 2021
"""

# import utils #nopep8
#utils.set_project_path()  # nopep8

import time
from importlib import reload
from termcolor import colored
from PIL import Image
import numpy as np


from qusoliton.sdenls import SDENLS3D as NLS

reload(NLS)
# %% start timing
STARTT = time.time()

# %% plot_level
PLOT_LEVEL = 2

# save data flag
SAVE_DATA = False
FILENAME = './data/testdata'

# load file for initial condition
FILENAME_RELAX = './data/relaxdata'

# %% define parameters
NPOINT = 32  # common point in any direction
NLS.xmin = -25
NLS.xmax = 25
NLS.nx = NPOINT
NLS.ymin = -25
NLS.ymax = 25
NLS.ny = NPOINT
NLS.tmin = -25
NLS.tmax = 25
NLS.nt = NPOINT
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
NLS.alphanoise = 1
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
W0X = 5.66  # initial waist
W0Y = 5.66  # initial waist
W0T = 5.66  # initial waist
A0 = 1.0
NLS.psi0 = A0*np.exp(-np.square(NLS.X/W0X) -
                     np.square(NLS.Y/W0Y)-np.square(NLS.T/W0T))
NLS.phi0 = np.copy(NLS.psi0)


# reload the data and evolve to test
print(colored('Loading and evolving ', 'blue'))
NLS.loadall(FILENAME_RELAX+'.npz')

# %% plot initial condition
if PLOT_LEVEL > 0:
    FIGURE_NAME = './img/initialguess.png'
    Image.open(NLS.plot_panels(
        NLS.psi0, FIGURE_NAME, 'initial condition')).show()


# evolve the state
NLS.alphanoise = 1
NLS.nphoton = 100
NLS.nshots = 3
NLS.filerootshot = './data/shot'
NLS.save_all_shots = True  # save all the shots in different files
NLS.init()  # reinit the module after changing parameters
OUTDATA = NLS.evolve_SDE_NLS(input)

if PLOT_LEVEL > 0:
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

# save the variables with the relaxation profile
if SAVE_DATA:
    NLS.saveall(FILENAME)
