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
Test the laplacian with fft2

@author: claudio
@created: 8 march 2020
@version: 6 february 2022
"""

# TODO: test the diffraction with a gaussian beam
import numpy as np
import matplotlib.pyplot as plt
from qusoliton.sdenls import SDENLS2D as NLS
# %% parameters (as a dictionary)
input = dict()
input['zmax'] = np.pi
input['xmin'] = -30.0
input['xmax'] = 30.0
input['ymin'] = -30.0
input['ymax'] = 30.0
input['nx'] = 256
input['ny'] = 256
input['cxx'] = 0.5
input['cyy'] = 2.0
# %% coordinates
x, y, _, _ = NLS.coordinates(input)
X, Y = np.meshgrid(x, y)
#  wavenumbers
kx, ky, \
    minus_kx_square, minus_ky_square = NLS.wavenumbers(x, y)
MKX2, MKY2 = np.meshgrid(minus_kx_square, minus_ky_square)

# laplacian
cxx = input['cxx']
cyy = input['cyy']

D2X = MKX2
D2Y = MKY2
LAPL = cxx*D2X+cyy*D2Y
# %% initial condition
psi0 = np.exp(-np.square(X)-np.square(Y))
input['psi0'] = psi0
# %% second derivative wrt x
psi0xx = np.real(np.fft.ifft2(D2X*np.fft.fft2(psi0)))

# %% second derivative wrt y
psi0yy = np.real(np.fft.ifft2(D2Y*np.fft.fft2(psi0)))

# %% evaluate laplacian
LAPL1 = np.real(np.fft.ifft2(LAPL*np.fft.fft2(psi0)))

# %%
fig2D = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, psi0)
fig2Db = plt.figure()
bx = plt.axes(projection='3d')
bx.plot_surface(Y, X, LAPL1)
# %% exact laplacal
psi0xxE = (-2.0+4.0*X*X)*psi0
psi0yyE = (-2.0+4.0*Y*Y)*psi0
LAPLE = cxx*psi0xxE+cyy*psi0yyE
# %%
fig2DE = plt.figure()
bx = plt.axes(projection='3d')
bx.plot_surface(Y, X, psi0xx)
# %%
fig2DE = plt.figure()
bx = plt.axes(projection='3d')
bx.plot_surface(Y, X, psi0yy)
# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
fig2Db = plt.figure()
ax1.pcolormesh(Y, X, np.real(LAPL1))
ax2.pcolormesh(Y, X, LAPLE)
# %%
errorLAPL = np.max(np.max(np.abs(LAPL1-LAPLE)))
print("ErrorLAPL "+repr(errorLAPL))
errorXX = np.max(np.max(np.abs(psi0xx-psi0xxE)))
print("ErrorYY "+repr(errorXX))
errorYY = np.max(np.max(np.abs(psi0yy-psi0yyE)))
print("ErrorXX "+repr(errorYY))
