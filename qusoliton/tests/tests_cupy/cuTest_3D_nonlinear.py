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
""" Created 3 jan 2022
"""

# set project path
#import utils
# utils.set_project_path()

# Import the os module
import time
import numpy as np
from qusoliton.cusdenls import cuSDENLS3D as NLS
from PIL import Image
#from importlib import reload
# import os
# set the path to the project root
# ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))


# def set_project_path():
#     """ se the path in the project root """
#     try:
#         os.chdir(ROOT_DIR)
#         print("Current working directory: {0}".format(os.getcwd()))
#     except FileNotFoundError:
#         print("Directory: {0} does not exist".format(ROOT_DIR))
#     except NotADirectoryError:
#         print("{0} is not a directory".format(ROOT_DIR))
#     except PermissionError:
#         print("You do not have permissions to change to {0}".format(ROOT_DIR))


# reload(NLS)
# %% start timing
STARTT = time.time()

# %% plot_level
PLOT_LEVEL = 1

# %% define parameters as dictionary

NPOINT = 16  # common point in any direction
NLS.xmin = -10
NLS.xmax = 10
NLS.nx = NPOINT
NLS.ymin = -10
NLS.ymax = 10
NLS.ny = NPOINT
NLS.tmin = -10
NLS.tmax = 10
NLS.nt = NPOINT
NLS.chi = 1.0
NLS.nplot = 10
NLS.nphoton = 100
NLS.nplot = 10
NLS.nz = 100
NLS.zmax = 1.0
NLS.verbose_level = 2
NLS.PLOT_LEVEL = 1
NLS.iter_implicit = 0
NLS.sigma_local = 1
NLS.sigma_nonlocal = 0
NLS.alphanoise = 0
# define main algol
NLS.step = NLS.DRUMMOND_step
NLS.iter_implicit = 3

# init module
NLS.init()

# initial condition
W0X = 2.0  # initial waist
W0Y = 2.0  # initial waist
W0T = 2.0  # initial waist
A0 = 5.0
PSI0 = A0*np.exp(-np.square(NLS.X/W0X) -
                 np.square(NLS.Y/W0Y)-np.square(NLS.T/W0T))
PHI0 = np.copy(PSI0)
NLS.psi0 = PSI0
NLS.phi0 = PHI0

# plot initial conditions

# %% plot initial condition
if PLOT_LEVEL > 0:
    FIGURE_NAME = './img/initialcondition.png'
    Image.open(NLS.plot_panels(PSI0, FIGURE_NAME, 'initial condition')).show()

# %% evolve
OUT = NLS.evolve_SDE_NLS(input)

# %% plot final state
Image.open(
    NLS.plot_panels(np.abs(NLS.psif), './img/currentplot.png',
                    'current status'
                    )).show()
Image.open(
    NLS.plot_observables(
        './img/observables.png',
        'observables'
    )).show()
