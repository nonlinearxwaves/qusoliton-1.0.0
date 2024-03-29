# This file is part of QuSoliton: Quantum Soliton toolbox.
#
# Copyright (c) 2022 and later, Claudio Conti.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:

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
Created on Thu Dec 26 18:22:38 2019

Solve the stochastic NLS by the HEUN method

The equation is written as

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi |psi|^2 psi

TEST 1D nonlinear
Test soliton propagation with initial condition N sech(x)
which is periodical with period pi/2 in t pi/2
with the following coefficients
chi = 0.5
cxx = 0.5

@author: claudio
@version: 06 july 2023
"""

# TODO: test the diffraction with a gaussian beam

import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS1D as NLS
import time

# %% timing
startt = time.time()
# %% parameters (as a dictionary)
input = dict()
input["zmax"] = np.pi
input["xmin"] = -5.0
input["xmax"] = 5.0
input["nx"] = 256
input["nplot"] = 50
input["nz"] = 100
input["cxx"] = 0.5
input["chi"] = 0.5
input["plot_level"] = 0
input["verbose_level"] = 2
# %% coordinates
x, _ = NLS.coordinates(input)
# %% initial condition
psi0 = 3 / np.cosh(x)
input["psi0"] = psi0
# %% evolve
out = NLS.evolve_NLS(input)
# %% extract data from sims
zplot = out["zplot"]
psi2D = out["psi2D"]
powers = out["powers"]
mean_xs = out["mean_xs"]
mean_square_xs = out["mean_square_xs"]
# %%
fig2D = plt.figure()
plt.pcolormesh(zplot, x, np.abs(psi2D))
plt.ylabel("x")
plt.xlabel("z")
# %% plot observables
plt.figure()
plt.plot(zplot, powers)
plt.ylabel("power")
plt.xlabel("z")
plt.figure()
plt.plot(zplot, mean_xs)
plt.ylabel("<x>")
plt.xlabel("z")
# %% plot the std and compare with exact solution
cxx = input["cxx"]
std_ex = (1.0 / 2.0) * np.sqrt(1.0 + 16.0 * cxx * (zplot) ** 2)
plt.figure()
plt.plot(zplot, np.sqrt(mean_square_xs))
plt.plot(zplot, std_ex, "rx")
plt.ylabel("sqrt(<x**2>)")
plt.xlabel("z")
plt.show()
# %% timing
endt = time.time()
if input["verbose_level"] > 0:
    print("Total time (s) " + repr(endt - startt))
