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
# test load relaxation data and evolve
""" Created 30 dic 2021
"""
#from importlib import reload
from qusoliton.sdenls import SDENLS3D as NLS
import time
from PIL import Image
import numpy as np
from termcolor import colored
#import utils
# utils.set_project_path()
# set project path

# need for debugging
# reload(NLS)

# %% start timing
startt = time.time()

# %% plot_level
plot_level = 1

# save data flag
filename_relax = './data/relaxdata'

# reload the data and evolve to test
print(colored('Loading and evolving ', 'blue'))
NLS.loadall(filename_relax+'.npz')

# %% plot
if plot_level > 0:
    Image.open(
        NLS.plot_panels(np.abs(NLS.psi0),
                        './img/relaxprofile.png',
                        'final state after relaxation'
                        )).show()

# evolve the state after turning on noise
NLS.chi = 1.0
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
