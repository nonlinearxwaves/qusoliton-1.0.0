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
"""Module for NLS stochastic.

Module with routines for the integration of the
Stochastic NLS

Deterministic version of the equation
-------------------------------------

1j psi_t + c_xx psi_xx + 2 chi |psi|^2 psi =0

psi_t = 1j c_xx psi_xx + 1j 2 chi |psi|^2 psi


Stochastic version of the equation
----------------------------------


psi_t = 1j c_xx psi_xx + 1j 2 chi psi^2 phi + noise_psi phi xi

phi_t = -1j c_xx psi_xx - 1j 2 chi psi phi^2 + noise_phi phi nu

with xi and nu stochastic fields with unitary values

Coefficient for the stochastic noise
------------------------------------

noise_psi=(- 1i  chi /n0 )
noise_phi=(- 1i chi /n0 )

with n0 the number of photons

The noise is scaled with sqrt(1/dx)


References
----------
Kloeden, Platen, _Numerical Solution of Stochastic Differential Equations_,
                Springer-Verlag
Greiner, Strimatter, and Honerkamp, JSP __51__, 95 (1988)
Honeycutt, PRA __45__, 600 (1992)
Sauer, Numerical Solution of SDE in Finance
Drummond ...

Versions
--------
Use cupy for CUDA run

Created on Apr 3, 2020

@author: claudio
@version: 3 april 2020
"""

# TODO: some routines can be optimize as evaluating 1j*2.0*chi and 1j*cxx
# and 0.5dz
#   to speed up
#
# also should preallocate phi1 and psi1 to speed up
#

# TODO: replace np.double and np.complex with dtreal and dtcomplex

# TODO: le variabili minux_kx_square e simili si possono mettere globali,
# in modo da evitare il message passing nelle chiamate a funzione

# TODO: improve numeric format in messages

import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cupy.fft as fft

# datatypes
cp_float = np.float32  # real datatype
cp_complex = np.complex  # complex datatype
np_float = np.float32  # real datatype
np_complex = np.complex  # complex datatype


def coordinates(input):
    """Return the coordinate x with a given input."""
    xmin = input['xmin']
    xmax = input['xmax']
    nx = input['nx']
    x = np.linspace(xmin, xmax, num=nx, dtype=cp_float)
    return x


def power(dx, psi, phi):
    """Return power of a field as the integral of psi*phi (in general it complex).

    Parameters
    ----------
    dx : increment
    psi : array of complex

    Returns
    -------
    The integral of |psi|**2 , if the field is normalized this is 1.0

    """
    return cp.sum(cp.real(psi*phi))*dx


def normalize(dx, psi, phi):
    """Return power of a field as the integral of |psi|^2.

    Parameters
    ----------
    x : array of reals
    psi : array of complex

    Returns
    -------
    normalize field psi
    power of the input field

    """
    P = power(dx, psi, phi)
    return psi/np.sqrt(P), P


def moments(x, psi, phi):
    """Return first moments wrt x of a field.

    Parameters
    ----------
    x : array of reals
    psi : array of complex

    Returns
    -------
    power : power of the beam
    mean_x : mean of x wrt |psi|^2
    mean_x_square : mean of x**2 wrt |psi|^2

    """

    # psi_norm, P = normalize(x, psi)
    # i_norm = np.abs(psi_norm)**2

    tmp = cp.real(cp.multiply(psi, phi))
    powtmp = cp.sum(tmp)
    mean_x = cp.sum(cp.multiply(x, tmp))/powtmp
    mean_x_square = cp.sum(cp.multiply(cp.square(x), tmp))/powtmp
    return powtmp, mean_x, mean_x_square


def EULER_step():
    """Return the deterministic rhs of SDE NLS by Euler-Maruyama method.

    References
    ----------
    Sauer

    Parameters
    ----------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(dz/dx)
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(dz/dx)
    and must be calculate befor calling this function

    Returns
    -------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

    Notes
    -----
    This tends to be unstable
    Can speed up by removing the random normal and using a uniform random,
        but this requires a different scaling for the noise


    """
    global dz, npsi, nphi, nx, psi, phi
    psi, phi = SDENLS_eq_fast()
    psi = psi + dz*psi + npsi*psi*cp.random.randn(nx)
    phi = phi + dz*phi + nphi*phi*cp.random.randn(nx)
    return


def MILSTEIN_step():
    """Return the deterministic rhs of SDE NLS by MILSTEIN method.

    The Milstein is strong scheme order O(1)

    References
    ----------
    Kloeden and Platen, section 10.3

    Parameters
    ----------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(dz/dx)
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(dz/dx)
    and must be calculate befor calling this function

    Returns
    -------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi

    Notes
    -----
    This tends to be unstable
    Can speed up by removing the random normal and using a uniform random


    """
    global dz, dzsquare, npsi, nphi, nx, halfnphisquare, halfnpsisquare
    global psi, phi
    psi, phi = SDENLS_eq_fast(psi, phi)
    DW1 = cp.random.randn(nx)
    DW2 = cp.random.randn(nx)
    DWpsi = npsi*DW1
    DWphi = nphi*DW2
    DWpsi += halfnpsisquare*(DW1**2-dzsquare)
    DWphi += halfnphisquare*(DW2**2-dzsquare)
    psi = psi + dz*psi + psi*DWpsi
    phi = phi + dz*phi + phi*DWphi
    return


def HEUN_step(psi, phi):
    """Return the deterministic rhs of SDE NLS via the HEUN algol.

    References
    ----------
    Greiner 1988

    Parameters
    ----------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/dx)
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/dx)
    and must be calculate befor calling this function

    Returns
    -------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """
    global dz, npsi, nphi
    Fpsi, Fphi = SDENLS_eq_fast(psi, phi)
    psi1 = psi + dz*Fpsi + npsi*psi*HEUN_rand()
    phi1 = phi + dz*Fphi + nphi*phi*HEUN_rand()
    Gpsi, Gphi = SDENLS_eq_fast(psi1, phi1)
    return \
        psi + 0.5*dz*(Fpsi + Gpsi) + npsi*psi*HEUN_rand(), \
        phi + 0.5*dz*(Fphi + Gphi) + nphi*phi*HEUN_rand()


def HEUN_rand():
    """Return a scaled random number for speed."""
    global nx
    h = np.random.rand(nx)
    for ix in range(nx):
        if h[ix] < ONE_SIX:
            h[ix] = 1.0
        elif h[ix] >= FIVE_SIX:
            h[ix] = -1.0
        else:
            h[ix] = 0.0
    # h[h < ONE_SIX] = 1.0
    # h[h >= FIVE_SIX] = -1.0
    # h[(h >= ONE_SIX) & (h < FIVE_SIX)] = 0.0
    return h


def evolve_SDE_NLS(input):
    """Evolve according to the NLS with an arbitrary algol.

    Parameters as input
    -------------------
    input is a dictionary encoding different parameters
    zmax = np.pi
    xmin = -30.0
    xmax = 30.0
    nx = 256
    nplot = 10
    nz = 10000
    cxx = 1.0
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

    # TODO: introdurre un dict di default (con config file) per i parametri
    # in input and also for the out

    start_time = time.time()

    cp.cuda.Device(0).use()

    # Extract parameter out of the dictionary
    zmax = input['zmax']
    nx = input['nx']
    nz = input['nz']
    nplot = input['nplot']
    cxx = input['cxx']
    chi = input['chi']
    n0 = input['n0']
    xmin = input['xmin']
    xmax = input['xmax']
    nx = input['nx']
    plot_level = input['plot_level']
    verbose_level = input['verbose_level']
    make_step = input['step']

    # additional variables
    twochi = 2.0*chi

    # coordinates
    x = cp.linspace(xmin, xmax, num=nx, dtype=cp_float)
    dx = x[1]-x[0]

    # wavenumbers
    kx = cp.zeros(x.shape, dtype=cp.float32)
    kx = 2.0*np.pi*cp.fft.fftfreq(nx, d=dx)
    minus_kx_square = -cp.square(kx)

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot
    dzsquare = dz**2  # used by Milstein algol

    # Set the scale for the noise depending on the chosen step algo
    noise_scale = np.sqrt(dz/dx)
    if make_step == HEUN_step:
        # scale for HEUN algol noise
        noise_scale = np.sqrt(3.0*dz/dx)
        if verbose_level > 1:
            print("HEUN algol chosen")
    elif make_step == EULER_step:
        # scale for EULER algol noise
        noise_scale = np.sqrt(dz/dx)
        if verbose_level > 1:
            print("EULER algol chosen")
    elif make_step == MILSTEIN_step:
        # scale for MILSTEIN algol noise
        noise_scale = np.sqrt(dz/dx)
        if verbose_level > 1:
            print("MILSTEIN algol chosen")
    else:
        print("ERROR no make step function specified")
        return

    # noise coefficients # TODO check noise coefficients!
    if input['noise']:
        chinoise = input['chi']
        if "chinoise" in input:  # check if a chinoise is defined
            chinoise = input['chinoise']
        npsi = cp.sqrt(2.0*1j*chinoise/n0)*noise_scale
        nphi = cp.sqrt(-2.0*1j*chinoise/n0)*noise_scale
        halfnpsisquare = (0.5*(npsi**2))  # used by Milstein algols
        halfnphisquare = (0.5*(nphi**2))  # used by Milstein algols
    else:
        npsi = 0.0
        nphi = 0.0
        halfnpsisquare = 0.0
        halfnphisquare = 0.0

    # vector of longitudinal points
    z = 0.0
    zplot = cp.zeros((nplot+1, ), dtype=np.float64)
    zplot[0] = z

    # open figure
    if plot_level > 1:
        figura = plt.figure(1)
        ax = figura.add_subplot(111)

    # allocate variables

    psi0cp = cp.asarray(input['psi0'])
    phi0cp = cp.asarray(input['phi0'])

 #   cp.cuda.Stream.null.synchronize()
    psi = cp.copy(psi0cp)
    phi = cp.copy(phi0cp)
    Fpsi = cp.zeros_like(psi0cp)
    Fphi = cp.zeros_like(phi0cp)
    psi_xx = cp.zeros_like(psi0cp)
    phi_xx = cp.zeros_like(phi0cp)
    psiphi = cp.zeros_like(phi0cp)

    #  store 2D matrices
    psi2D = cp.zeros((nx, nplot+1), dtype=np.complex64)
    psi2D[:, 0] = psi0cp
    phi2D = cp.zeros_like(psi2D)
    phi2D[:, 0] = phi0cp

    # store observable quantities
    powers = cp.zeros(zplot.shape, dtype=np.float64)
    mean_xs = cp.zeros_like(powers)
    mean_square_xs = cp.zeros_like(mean_xs)

    # initial values for the observables
    powers[0], mean_xs[0], mean_square_xs[0] = moments(x, psi0cp, phi0cp)


# main loop
    for iplot in range(nplot):
        with cp.cuda.profile():
            for iz in range(nz):
                # deterministic part
                psi_xx = cp.fft.ifft(cp.multiply(
                    minus_kx_square, cp.fft.fft(psi)))
                phi_xx = cp.fft.ifft(cp.multiply(
                    minus_kx_square, cp.fft.fft(phi)))
                psiphi = cp.multiply(psi, phi)
                Fpsi = cp.multiply(psi, psiphi)
                Fphi = cp.multiply(phi, psiphi)
                Fpsi = +1j*cxx*psi_xx+1j*twochi*Fpsi
                Fphi = -1j*cxx*phi_xx-1j*twochi*Fphi
            # stochastic part
                DW1 = cp.random.randn(nx)
                DW2 = cp.random.randn(nx)
                DWpsi = npsi*DW1
                DWphi = nphi*DW2
                DWpsi += halfnpsisquare*(DW1**2-dzsquare)
                DWphi += halfnphisquare*(DW2**2-dzsquare)
                psi = psi + dz*Fpsi + psi*DWpsi
                phi = phi + dz*Fphi + phi*DWphi
                z = z+dz
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot "
                  + repr(iplot+1)+" of "+repr(nplot))
        if plot_level > 1:
            plt.figure(1)
            plt.plot(x.get(), cp.abs(psi0cp).get(), 'k+')
            plt.plot(x.get(), cp.abs(psi).get())
            plt.title(repr(iplot)+' z='+repr(z))
            plt.xlabel('x')
            figura.canvas.draw()
            figura.canvas.flush_events()
        #            plt.show()
        # store
        psi2D[:, iplot+1] = psi
        phi2D[:, iplot+1] = phi
        zplot[iplot+1] = z
        powers[iplot+1], \
            mean_xs[iplot+1], \
            mean_square_xs[iplot+1] = moments(x, psi, phi)

    # timing
    end_time = time.time()
    total_time = end_time-start_time

    # store output e retransfer from the main device
    out = dict()
    out['input'] = input
    out['zplot'] = cp.asnumpy(zplot)
    out['powers'] = cp.asnumpy(powers)
    out['psi2D'] = cp.asnumpy(psi2D)
    out['phi2D'] = cp.asnumpy(phi2D)
    out['mean_xs'] = cp.asnumpy(mean_xs)
    out['mean_square_xs'] = cp.asnumpy(mean_square_xs)
    out['time_seconds'] = total_time

    if verbose_level > 0:
        print("Run time (seconds) %6.2f " % (total_time))

    # Return
    return out
