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
""" Module for NLS stochastic

Module with routines for the integration of the
Stochastic NLS

-------------------------------
Deterministic version of the equation

1j psi_t + c_xx psi_xx + c_yy psi_yy + 2 chi |psi|^2 psi =0

psi_t = 1j c_xx psi_xx+ 1j c_yy psi_yy + 1j 2 chi |psi|^2 psi


-------------------------------
Stochastic version of the equation


psi_t = 1j c_xx psi_xx + 1j c_yy psi_yy + 1j 2 chi psi^2 phi + noise_psi phi xi

phi_t = -1j c_xx phi_xx-1j c_yy phi_yy - 1j 2 chi psi phi^2 + noise_phi phi nu

with xi and nu stochastic fields with unitary values

-------------------------------
Coefficient for the stochastic noise

noise_psi=(- 1i 2.0 chi /n0 )
noise_phi=(- 1i 2.0 chi /n0 )

with n0 the number of photons

References
----------
Drummond ...

File history
------------

Created on Sun Dec 29 19:39:43 2019

@author: claudio
@version:9 march 2020
"""

# TODO: optimize 1j*cxx as 1jcxx

import time
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype

# datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype

# global variables
nx = 2
ny = 2
cxx = 1.0
cyy = 1.0
chi = 1.0
twochi = 2.0
dz = 0.1
dzsquare = 0.01
npsi = 0.0
nphi = 0.0
halfnpsisquare = 0.0
halfnphisquare = 0.0
LAPL = np.zeros((ny, nx), dtype=dtreal)  # laplacian operator
# costant for random number generation
ONE_SIX = 0.166666666666666665
FIVE_SIX = 0.8333333333333334


def wavenumbers(x, y):
    """ Return the wavenumbers for fft corresponding to x

    Parameters
    ----------------------
    x: array of doubles, must have shape (nx,)
    y: array of doubles, must have shape (ny,)

    Returns
    ----------------------
    kx  : vector of wavenumbers, shape (nx, )
    minus_kx_square : -kx**2 for second derivative, shape (nx, )
    ky  : vector of wavenumbers, shape (ny, )
    minus_ky_square : -ky**2 for second derivative, shape (ny, )

    """
    nx = x.shape[0]
    dx = x[1]-x[0]
    kx = np.zeros(x.shape, dtype=dtreal)
    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
    ny = y.shape[0]
    dy = y[1]-y[0]
    ky = np.zeros(y.shape, dtype=dtreal)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dy)
    return kx, ky, -kx**2, -ky**2


def peak(psi, phi):
    """ Return power of a field as the integral of |psi|^2

    Parameters
    --------------
    psi : array of complex
    dx : increment along x
    dy : increment along y

    Returns
    --------------
    The integral of |psi|**2 , if the field is normalized this is 1.0

    """
    intensity = np.real(psi*phi)
    return np.max(np.max(intensity))


def power(psi, phi, dxdy=1.0):
    """ Return power of a field as the integral of real(psi*phi)

    Parameters
    --------------
    psi : array of complex
    dx : increment along x
    dy : increment along y

    Returns
    --------------
    The integral of |psi|**2 , if the field is normalized this is 1.0

    """
    intensity = np.real(psi*phi)
    return np.sum(np.sum(intensity))*dxdy


def normalize(psi, phi, dxdy=1.0):
    """ Return a field normalized to have unitary power as phi*psi

    Parameters
    --------------
    x : array of reals
    psi : array of complex

    Returns
    --------------
    normalize field psi
    power of the input field

    """
    P = power(psi, phi, dxdy)
    return psi/np.sqrt(P), phi/np.sqrt(P), P


def moments(X, Y, psi, phi, dxdy=1.0):
    """ Return first moments wrt x of a field (psi, phi)

    Parameters
    --------------
    X : 2D array (meshgrid) of reals
    Y : 2D array (meshgrid) of reals
    psi : array of complex

    Returns
    --------------
    power : power of the beam
    mean_x : mean of x wrt |psi|^2
    mean_y : mean of x wrt |psi|^2
    mean_x2 : mean of x**2 wrt |psi|^2
    mean_y2 : mean of x**2 wrt |psi|^2
    mean_r2 : mean of x**2 + y**2 wrt |psi|^2

    """
    psi_norm, phi_norm, P = normalize(psi, phi, dxdy)
    i_norm = np.real(psi_norm*phi_norm)
    mean_x = np.sum(X*i_norm)*dxdy
    mean_y = np.sum(Y*i_norm)*dxdy
    mean_x2 = np.sum((X**2)*i_norm)*dxdy
    mean_y2 = np.sum((Y**2)*i_norm)*dxdy
    mean_r2 = mean_x2+mean_y2
    return P, mean_x, mean_y, mean_x2, mean_y2, mean_r2


# %% costant for random number generation
ONE_SIX = 0.166666666666666665
FIVE_SIX = 0.8333333333333334


# %%
def coordinates(input):
    """ Return the coordinate x, y with a given input """
    xmin = input['xmin']
    xmax = input['xmax']
    nx = input['nx']
    x = np.linspace(xmin, xmax, num=nx, dtype=dtreal)
    dx = x[1]-x[0]
    ymin = input['ymin']
    ymax = input['ymax']
    ny = input['ny']
    y = np.linspace(ymin, ymax, num=ny, dtype=dtreal)
    dy = y[1]-y[0]
    return x, y, dx, dy


# %%
# # def evolve_SDE_NLS_Heun(input):
#     """ Evolve according to the NLS teterministic

#     Parameters as input
#     -------------------
#     input is a dictionary encoding different parameters
#     zmax = np.pi
#     xmin = -30.0
#     xmax = 30.0
#     ymin = -30.0
#     ymax = 30.0
#     nx = 256
#     ny = 256
#     nplot = 10
#     nz = 10000
#     cxx = 1.0
#     cyy = 1.0
#     chi = 0.0
#     n0 = 10000
#     plot_level=1

#     Returns
#     ------------------------
#     A dictionary out with various output

#     """

#     # Extract parameter out of the dictionary
#     zmax = input['zmax']
#     nx = input['nx']
#     ny = input['ny']
#     nz = input['nz']
#     nplot = input['nplot']
#     cxx = input['cxx']
#     cyy = input['cyy']
#     chi = input['chi']
#     n0 = input['n0']
#     plot_level = input['plot_level']
#     verbose_level = input['verbose_level']

#     # coordinates
#     x, y, dx, dy = coordinates(input)
#     dxdy = dx*dy

#     # initial condition
#     psi0 = input['psi0']
#     phi0 = input['phi0']

#     #  wavenumbers
#     [kx, ky, minus_kx_square, minus_ky_square] = wavenumbers(x)
#     # TODO: minus_kx_square and minus_ky_square not needed

#     # meshgrids
#     X, Y = np.meshgrid(x, y)
#     KX, KY = np.meshgrid(kx, ky)
#     # TODO: test meshgrids

#     # laplacian
#     LAPL = -cxx*KX**2-cyy*KY**2

#     # longitudinal step (CHECK HERE)
#     dz = (zmax / nz) / nplot
#     halfdz = 0.5*dz

#     # noise coefficients
#     if input['noise']:
#         npsi = np.sqrt(1j*2.0*chi/n0)*np.sqrt(3.0*dz/dxdy)
#         nphi = np.sqrt(-1j*2.0*chi/n0)*np.sqrt(3.0*dz/dxdy)
#     else:
#         npsi = 0.0
#         nphi = 0.0

#     # vector of longitudinal points
#     z = 0.0
#     zplot = np.zeros((nplot+1, ), dtype=dtreal)
#     zplot[0] = z

#     #  store 3D matrices
#     psi3D = np.zeros((nx, ny, nplot+1), dtype=dtcomplex)
#     psi3D[:, :, 0] = psi0
#     phi3D = np.zeros((nx, ny, nplot+1), dtype=dtcomplex)
#     phi3D[:, :, 0] = psi0

#     # store observable quantities
#     powers = np.zeros(zplot.shape, dtype=dtreal)
#     mean_xs = np.zeros(zplot.shape, dtype=dtreal)
#     mean_ys = np.zeros(zplot.shape, dtype=dtreal)
#     mean_x2s = np.zeros(zplot.shape, dtype=dtreal)
#     mean_y2s = np.zeros(zplot.shape, dtype=dtreal)
#     mean_r2s = np.zeros(zplot.shape, dtype=dtreal)

#     # initial values for the observables
#     powers[0], mean_xs[0], mean_ys[0], \
#         mean_x2s[0], mean_y2s[0], \
#         mean_r2s[0] = moments(X, Y, psi0, dxdy)

#     # %% laplacian function
#     def d_xx_yy(psi):
#         """ Return the second derivative of the field psi by fft

#         Parameters
#         --------------
#         psi : array of complex64 for the field

#         Returns
#         --------------
#         cxx psi_xx+ cyy psi_yy : second derivatives with respect to x

#         """
#         nonlocal LAPL
#         return fft.ifft2(LAPL*fft.fft2(psi))

#     # %% main equations
#     def SDENLS_eq_fast(psi, phi):
#         """ Return the deterministic rhs of SDE NLS

#         Parameters
#         --------------
#         psi : array of complex64 for the field psi
#         phi : array of complex64 for the field phi

#         Returns
#         --------------
#         Two output complex vectors

#         +1j c_xx psi_xx+c_yy psi_yy + 1j 2.0 chi psi^2 phi
#         -1j c_xx psi_xx+c_yy psi_yy - 1j 2.0 chi psi phi^2


#         """
#         nonlocal chi
#         psi_xx_yy = d_xx_yy(psi)
#         phi_xx_yy = d_xx_yy(phi)
#         tmp = psi*phi
#         psi_ = +1j*psi_xx_yy+1j*2.0*chi*psi*tmp
#         phi_ = -1j*phi_xx_yy-1j*2.0*chi*phi*tmp
#         return psi_, phi_

#     # %% random number
#     def HEUN_rand():
#         """ Return a scaled random number for speed """
#         h = np.random.rand(nx, ny)
#         h[h < ONE_SIX] = 1.0
#         h[h >= FIVE_SIX] = 1.0
#         h[(h >= ONE_SIX) & (h < FIVE_SIX)] = 0.0
#         return h

#     # %% define main function
#     def HEUN_step_fast(minus_kx_square, psi, phi,
#                        npsi, nphi, dz, nx,
#                        cxx=1.0, chi=1.0):
#         """ Return the deterministic rhs of SDE NLS

#         Parameters
#         --------------
#         psi : array of complex64 for the field psi
#         phi : array of complex64 for the field phi

#         IMPORTANT, for reason of speed the noise coefficients are defined as
#         npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
#         nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
#         and must be calculate befor calling this function

#         Returns
#         --------------
#         Two next update of the equations with the model

#         +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
#         -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


#         """
#         global dz, halfdz, npsi, nphi
#         Fpsi, Fphi = SDENLS_eq_fast(psi, phi)
#         psi1 = psi + dz*Fpsi + npsi*psi*HEUN_rand()
#         phi1 = phi + dz*Fphi + nphi*phi*HEUN_rand()
#         Gpsi, Gphi = SDENLS_eq_fast(psi1, phi1)
#         psi_ = psi + halfdz*(Fpsi + Gpsi) + npsi*psi*HEUN_rand()
#         phi_ = phi + halfdz*(Fphi + Gphi) + nphi*phi*HEUN_rand()
#         return psi_, phi_

#     # %% main loop
#     # open figure
#     if plot_level > 0:
#         plt.figure(1)
#     # main loop
#     psi = psi0
#     phi = phi0
#     for iplot in range(nplot):
#         for iz in range(nz):
#             psi, phi = \
#                 HEUN_step_fast(psi, phi)
#             z = z+dz
#         # temporary current field solution and initial one
#         if verbose_level > 1:
#             print("Current plot "
#                   + repr(iplot+1)+" of "+repr(nplot))
#         if plot_level > 0:
#             plt.figure(1)
#             plt.pcolormesh(np.abs(psi))
#             plt.title(repr(iplot)+' z='+repr(z))
#             plt.show()
#         # store
#         iplot1 = iplot+1
#         psi3D[:, :, iplot1] = psi
#         phi3D[:, :, iplot1] = phi
#         zplot[iplot1] = z
#         powers[iplot1], mean_xs[iplot1], mean_ys[iplot1], \
#             mean_x2s[iplot1], mean_y2s[iplot1], \
#             mean_r2s[iplot1] = moments(X, Y, psi, dxdy)

#     # store output
#     out = dict()
#     out['input'] = input
#     out['zplot'] = zplot
#     out['powers'] = powers
#     out['psi3D'] = psi3D
#     out['phi3D'] = phi3D
#     out['mean_xs'] = mean_xs
#     out['mean_ys'] = mean_xs
#     out['mean_x2s'] = mean_x2s
#     out['mean_y2s'] = mean_y2s
#     out['mean_r2s'] = mean_r2s

#     # Return
#     return out


# %% laplacian function
def d_xx_yy(psi):
    """ Return the second derivative of the field psi by fft

    Parameters
    --------------
    psi : array of complex64 for the field

    Returns
    --------------
    cxx psi_xx+ cyy psi_yy : second derivatives with respect to x

    """
    global LAPL
    return fft.ifft2(LAPL*fft.fft2(psi))
#    return fft.ifft2(LAPL*fft.fft2(psi))/(nx*ny)


# %% main equations
def SDENLS_eq(psi, phi):
    """ Return the deterministic rhs of SDE NLS

    Parameters
    --------------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    Returns
    --------------
    Two output complex vectors

    +1j c_xx psi_xx+c_yy psi_yy + 1j 2.0 chi psi^2 phi
    -1j c_xx psi_xx+c_yy psi_yy - 1j 2.0 chi psi phi^2


    """
    global chi
    psi_xx_yy = d_xx_yy(psi)
    phi_xx_yy = d_xx_yy(phi)
    tmp = psi*phi
    psi_ = +1j*psi_xx_yy+1j*2.0*chi*psi*tmp
    phi_ = -1j*phi_xx_yy-1j*2.0*chi*phi*tmp
    return psi_, phi_


# %% define main function for the HEUN step
def HEUN_step(psi, phi):
    """Make a step with HEUN algol

    Parameters
    --------------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    and must be calculate befor calling this function

    Returns
    --------------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """
    global dz, halfdz, npsi, nphi
    Fpsi, Fphi = SDENLS_eq(psi, phi)
    psi1 = psi + dz*Fpsi + npsi*psi*HEUN_rand()
    phi1 = phi + dz*Fphi + nphi*phi*HEUN_rand()
    Gpsi, Gphi = SDENLS_eq(psi1, phi1)
    psi_ = psi + halfdz*(Fpsi + Gpsi) + npsi*psi*HEUN_rand()
    phi_ = phi + halfdz*(Fphi + Gphi) + nphi*phi*HEUN_rand()
    return psi_, phi_


def HEUN_rand():
    """Return a scaled random number for speed."""
    global nx, ny
    h = np.random.rand(nx, ny)
    for ix in range(nx):
        for iy in range(ny):
            if h[ix, iy] < ONE_SIX:
                h[ix, iy] = 1.0
            elif h[ix, iy] >= FIVE_SIX:
                h[ix, iy] = -1.0
            else:
                h[ix, iy] = 0.0
    # h[h < ONE_SIX] = 1.0
    # h[h >= FIVE_SIX] = -1.0
    # h[(h >= ONE_SIX) & (h < FIVE_SIX)] = 0.0
    return h


# %% define main function for EULER step
def EULER_step(psi, phi):
    """Make a step with EULER algol

    Parameters
    --------------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    and must be calculate befor calling this function

    Returns
    --------------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """
    global dz, npsi, nphi, nx, ny
    Fpsi, Fphi = SDENLS_eq(psi, phi)
    psi1 = psi + dz*Fpsi + npsi*psi*np.random.normal(size=(ny, nx))
    phi1 = phi + dz*Fphi + nphi*phi*np.random.normal(size=(ny, nx))
    return psi1, phi1


# %% define main function for MILSTEIN step
def MILSTEIN_step(psi, phi):
    """Make a step with MILSTEIN algol

    Parameters
    --------------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    and must be calculate befor calling this function

    Returns
    --------------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """
    global dz, npsi, nphi, nx, ny
    Fpsi, Fphi = SDENLS_eq(psi, phi)
    DW1 = np.random.normal(size=(ny, nx))
    DW2 = np.random.normal(size=(ny, nx))
    DWpsi = npsi*DW1
    DWphi = nphi*DW2
    DWpsi += halfnpsisquare*(DW1**2-dzsquare)
    DWphi += halfnphisquare*(DW2**2-dzsquare)
    psi1 = psi + dz*Fpsi + psi*DWpsi
    phi1 = phi + dz*Fphi + phi*DWphi
    return psi1, phi1


# %% main evolution routine
def evolve_SDE_NLS(input):
    """Evolve according to the NLS with an arbitrary algol.

    Parameters as input
    -------------------
    input is a dictionary encoding different parameters
    zmax = np.pi
    xmin = -30.0
    xmax = 30.0
    ymin = -30.0
    ymax = 30.0
    nx = 256
    ny = 256
    nplot = 10
    nz = 10000
    cxx = 1.0
    cyy = 1.0
    chi = 0.0
    n0 = 10000
    plot_level=1
    evolver = name of function for the evolving step

    plot_level = 2 plot intermediate fields

    Returns
    -------
    A dictionary out with various output

    """
    # TODO: detail the output and input in comments

    # TODO: introdurre un dict di default per i parametri
    # in input and also for the out

    start_time = time.time()
    global LAPL, \
        dz, dzsquare, \
        npsi, nphi, \
        halfnpsisquare, halfnphisquare, \
        nx, ny, chi, twochi, cxx, cyy

    # Extract parameter out of the dictionary
    zmax = input['zmax']
    nx = input['nx']
    ny = input['ny']
    nz = input['nz']
    nplot = input['nplot']
    cxx = input['cxx']
    cyy = input['cyy']
    chi = input['chi']
    n0 = input['n0']
    plot_level = input['plot_level']
    verbose_level = input['verbose_level']
    make_step = input['step']

    # additional variables
    twochi = 2.0*chi

    # coordinates
    x, y, dx, dy = coordinates(input)
    X, Y = np.meshgrid(x, y)

    # initial condition
    psi0 = input['psi0']
    phi0 = input['phi0']

    #  wavenumbers
    kx, ky, \
        minus_kx_square, minus_ky_square = wavenumbers(x, y)
    MKX2, MKY2 = np.meshgrid(minus_kx_square, minus_ky_square)

    # laplacian
    LAPL = cxx*MKX2+cyy*MKY2

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot
    dzsquare = dz**2  # used by Milstein algol

    # Set the scale for the noise depending on the chosen step algo
    if make_step == HEUN_step:
        # scale for HEUN algol noise
        noise_scale = np.sqrt(3.0*dz/(dx*dy))
        if verbose_level > 1:
            print("HEUN algol chosen")
    elif make_step == EULER_step:
        # scale for EULER algol noise
        noise_scale = np.sqrt(dz/(dx*dy))
        if verbose_level > 1:
            print("EULER algol chosen")
    elif make_step == MILSTEIN_step:
        # scale for MILSTEIN algol noise
        noise_scale = np.sqrt(dz/(dx*dy))
        if verbose_level > 1:
            print("MILSTEIN algol chosen")
    else:
        print("ERROR no make step function specified")
        return

    # noise coefficients
    if input['noise']:
        chinoise = input['chi']
        if "chinoise" in input:  # check if a chinoise is defined
            chinoise = input['chinoise']
        npsi = np.sqrt(1j*2.0*chinoise/n0)*noise_scale
        nphi = np.sqrt(-1j*2.0*chinoise/n0)*noise_scale
        halfnpsisquare = 0.5*(npsi**2)  # used by Milstein algols
        halfnphisquare = 0.5*(nphi**2)  # used by Milstein algols
    else:
        npsi = 0.0
        nphi = 0.0
        halfnpsisquare = 0.0
        halfnphisquare = 0.0

    # vector of longitudinal points
    z = 0.0
    zplot = np.zeros((nplot+1, ), dtype=np.double)
    zplot[0] = z

    #  store 2D matrices
    psi3D = np.zeros((ny, nx, nplot+1), dtype=np.complex64)
    psi3D[:, :, 0] = psi0
    phi3D = np.zeros((ny, nx, nplot+1), dtype=np.complex64)
    phi3D[:, :, 0] = phi0

    # store observable quantities
    peaks = np.zeros(zplot.shape, dtype=np.double)
    powers = np.zeros(zplot.shape, dtype=np.double)
    mean_xs = np.zeros(zplot.shape, dtype=np.double)
    mean_ys = np.zeros(zplot.shape, dtype=np.double)
    mean_square_xs = np.zeros(zplot.shape, dtype=np.double)
    mean_square_ys = np.zeros(zplot.shape, dtype=np.double)
    mean_square_rs = np.zeros(zplot.shape, dtype=np.double)

    # initial values for the observables
    peaks[0] = peak(psi0, phi0)
    powers[0], mean_xs[0], mean_ys[0], \
        mean_square_xs[0], mean_square_ys[0], \
        mean_square_rs[0] = moments(X, Y, psi0, phi0)

    # open figure
    if plot_level > 1:
        plt.figure(1)
    # main loop
    psi = psi0
    phi = phi0
    for iplot in range(nplot):
        for iz in range(nz):
            psi, phi = \
                make_step(psi, phi)
            z = z+dz
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot %d of %d, power = %5.2f "
                  % (iplot+1, nplot, powers[iplot]))
        if plot_level > 1:
            plt.figure(1)
            # plt.plot(x, np.abs(psi0), 'k+')
            plt.pcolor(Y, X, np.abs(psi))
            plt.title(repr(iplot)+' z='+repr(z))
            plt.xlabel('y')
            plt.ylabel('x')
            plt.show()
        # store
        psi3D[:, :, iplot+1] = psi
        phi3D[:, :, iplot+1] = phi
        zplot[iplot+1] = z
        powers[iplot+1], \
            mean_xs[iplot+1], \
            mean_ys[iplot+1], \
            mean_square_xs[iplot+1], \
            mean_square_ys[iplot+1], \
            mean_square_rs[iplot+1] = moments(X, Y, psi, phi)
        peaks[iplot+1] = peak(psi, phi)

    # timing
    end_time = time.time()
    total_time = end_time-start_time

    # store output (improve the output for 3D or similar)
    out = dict()
    out['input'] = input
    out['zplot'] = zplot
    out['powers'] = powers
    out['psi3D'] = psi3D
    out['phi3D'] = phi3D
    out['peaks'] = peaks
    out['mean_xs'] = mean_xs
    out['mean_ys'] = mean_ys
    out['mean_square_xs'] = mean_square_xs
    out['mean_square_ys'] = mean_square_ys
    out['mean_square_rs'] = mean_square_rs
    out['time_seconds'] = total_time

    if verbose_level > 0:
        print("Run time (seconds) %6.2f " % (total_time))

    # Return
    return out
