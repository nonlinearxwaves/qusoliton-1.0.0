# -*- coding: utf-8 -*-
# test load relaxation data and evolve
""" Created 3 jan 2021
"""

import utils
utils.set_project_path()


from termcolor import colored
import numpy as np
from PIL import Image
import time
import cupySDENLS.cuSDENLS3D as NLS
# set project path

# need for debugging
from importlib import reload
reload(NLS)

# %% start timing
startt = time.time()

# %% plot_level
plot_level = 1

# save data flag
filename_relax = './test_cupy/data/relaxdata'

# reload the data and evolve to test
print(colored('Loading and evolving ', 'blue'))
NLS.loadall(filename_relax+'.npz')

# %% plot
if plot_level > 0:
    Image.open(
        NLS.plot_panels(np.abs(NLS.psi0),
                        './test_cupy/img/relaxprofile.png',
                        'final state after relaxation'
                        )).show()

# evolve the state after turning on noise
NLS.chi = 1.0
NLS.init()  # reinit module after changing parameters
outdata = NLS.evolve_SDE_NLS(input)

if plot_level > 0:
    Image.open(
        NLS.plot_panels(np.abs(NLS.psif),
                        './test_cupy/img/finalepsi.png',
                        'final |psi| after evolution'
                        )).show()
    Image.open(
        NLS.plot_observables(
            './test_cupy/img/observables.png',
            'observables'
        )).show()
