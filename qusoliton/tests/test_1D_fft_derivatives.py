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
Created on Thu Dec 26 17:47:31 2019

Studies for different SDE algols for stochastic NLS1D

Here we study fft and second derivatives with numpy

@author: claudio
@version: 6 february 2022
"""

import numpy as np
import matplotlib.pyplot as plt


# %% parameters
nx = 128
xmin = -5.0
xmax = 5.0
# %% vector of points in the x-space
x = np.linspace(xmin, xmax, num=nx)
dx = x[1]-x[0]
# %% initial wavefunction
psi = np.exp(-np.square(x), dtype=np.complex64)
# %% first derivative of initial wavefunction
psi_x_ex = -2*x*psi
# %% second derivative of initial wavefunction
psi_xx_ex = -2*psi+4*x*x*psi
# %% plot initial wavefunction
plt.figure
plt.plot(x, np.abs(psi))
plt.xlabel('x')
plt.ylabel('\\psi')
# %% vector of wavenumbers
kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
# %% fft
fftpsi = np.fft.fft(psi)
plt.figure
plt.plot(np.fft.fftshift(kx), np.fft.fftshift(np.abs(fftpsi)))
plt.xlabel('kx')
plt.ylabel('fft \\psi')

# %% first derivative
psi_x = np.fft.ifft(1j*kx*fftpsi)
fig = plt.figure()
plt.plot(x, np.real(psi_x))
plt.plot(x, np.imag(psi_x), 'y')
plt.plot(x, np.real(psi_x_ex), 'rx')
plt.xlabel('x')
plt.ylabel('d_x \\psi')

# %% second derivative
psi_xx = np.fft.ifft(-kx*kx*fftpsi)
fig = plt.figure()
plt.plot(x, np.real(psi_xx))
plt.plot(x, np.imag(psi_xx), 'y')
plt.plot(x, np.real(psi_xx_ex), 'rx')
plt.xlabel('x')
plt.ylabel('d_xx \\psi')
plt.show()
