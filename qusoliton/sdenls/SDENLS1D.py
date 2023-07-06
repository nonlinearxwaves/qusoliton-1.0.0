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
Created on Sun Dec 29 19:39:43 2019

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
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype

# global variables
nx = 2
minus_kx_square = np.zeros((nx,))
conv = np.zeros((nx,), dtype=dtcomplex)
cxx = 1.0
chi = 1.0
twochi = 2.0
twochi1j = 1j
dz = 0.1
dzsquare = 0.01
npsi = 0.0
nphi = 0.0
halfnpsisquare = 0.0
halfnphisquare = 0.0
sqrt12dz = 0.1  # used for Euler integrator speed up

# costant for random number generation
ONE_SIX = 0.166666666666666665
FIVE_SIX = 0.8333333333333334


def wavenumbers(x):
    """Return the wavenumbers for fft corresponding to x.

    Parameters
    ----------------------
    x: array of doubles, must have shape (nx,)

    Returns
    ----------------------
    kx  : vector of wavenumbers
    minus_kx_square : -kx**2 for second derivative

    """
    nx = x.shape[0]
    dx = x[1] - x[0]
    kx = np.zeros(x.shape, dtype=dtreal)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    return kx, -(kx**2)


def d_xx(x, psi):
    """Return the second derivative of the field psi by fft.

    Parameters
    ----------
    x   : array of doubles of size nx
    psi : array of complex64 for the field


    Returns
    -------
    second derivatives with respect to x


    """
    # evaluate the wavenumbers
    kx, minus_kx_square = wavenumbers(x)
    return fft.ifft(minus_kx_square * fft.fft(psi))


def d_xx_fast(psi):
    """Return the second derivative of the field psi by fft.

    Parameters
    ----------
    minus_kx_square  : array of doubles wavenumbers for fft corresponding to
        to the squares of kx with minus sign -kx**2
        (may be calculated by the wavenumbers method)
    psi : array of complex64 for the field

    Returns
    -------
    psi_xx : second derivatives with respect to x

    """
    global minus_kx_square
    return fft.ifft(minus_kx_square * fft.fft(psi))


def NLS_eq(x, psi, cxx=1.0, chi=1.0):
    """Return rhs of deterministic NLS."""
    psi_xx = d_xx(x, psi)
    return 1j * cxx * psi_xx + 1j * 2.0 * chi * psi * (np.abs(psi) ** 2)


def NLS_eq_fast(psi):
    """Return rhs of deterministic NLS."""
    global chi, cxx
    psi_xx = d_xx_fast(psi)
    return 1j * cxx * psi_xx + 1j * 2.0 * chi * psi * (np.abs(psi) ** 2)


def SDENLS_eq_fast(psi, phi):
    """Return the deterministic rhs of SDE NLS.

    Parameters
    ----------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    Returns
    -------
    Two output complex vectors

    +1j c_xx psi_xx + 1j 2.0 chi psi^2 phi
    -1j c_xx psi_xx - 1j 2.0 chi psi phi^2


    """
    global twochi, cxx
    psi_xx = d_xx_fast(psi)
    phi_xx = d_xx_fast(phi)
    tmp = psi * phi
    return (
        +1j * cxx * psi_xx + 1j * twochi * psi * tmp,
        -1j * cxx * phi_xx - 1j * twochi * phi * tmp,
    )


def power(x, psi, phi):
    """Return power of a field as the integral of psi*phi (in general it complex).

    Parameters
    ----------
    x : array of reals
    psi : array of complex

    Returns
    -------
    The integral of |psi|**2 , if the field is normalized this is 1.0

    """
    return np.trapz(np.real(psi * phi), x)


def normalize(x, psi, phi):
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
    P = power(x, psi, phi)
    return psi / np.sqrt(P), P


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

    tmp = np.real(psi * phi)
    powtmp = np.trapz(tmp, x)
    mean_x = np.trapz(x * tmp, x) / powtmp
    mean_x_square = np.trapz((x**2) * tmp, x) / powtmp
    return powtmp, mean_x, mean_x_square


def EULER_step(psi, phi):
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
    global dz, npsi, nphi, nx
    Fpsi, Fphi = SDENLS_eq_fast(psi, phi)
    psi1 = psi + dz * Fpsi + npsi * psi * np.random.normal(size=nx)
    phi1 = phi + dz * Fphi + nphi * phi * np.random.normal(size=nx)
    return psi1, phi1


def MILSTEIN_step(psi, phi):
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
    Fpsi, Fphi = SDENLS_eq_fast(psi, phi)
    DW1 = np.random.normal(size=nx)
    DW2 = np.random.normal(size=nx)
    DWpsi = npsi * DW1
    DWphi = nphi * DW2
    DWpsi += halfnpsisquare * (DW1**2 - dzsquare)
    DWphi += halfnphisquare * (DW2**2 - dzsquare)
    psi1 = psi + dz * Fpsi + psi * DWpsi
    phi1 = phi + dz * Fphi + phi * DWphi
    return psi1, phi1


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
    psi1 = psi + dz * Fpsi + npsi * psi * HEUN_rand()
    phi1 = phi + dz * Fphi + nphi * phi * HEUN_rand()
    Gpsi, Gphi = SDENLS_eq_fast(psi1, phi1)
    return (
        psi + 0.5 * dz * (Fpsi + Gpsi) + npsi * psi * HEUN_rand(),
        phi + 0.5 * dz * (Fphi + Gphi) + nphi * phi * HEUN_rand(),
    )


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


def SRKII_step(psi, phi):
    """Return the deterministic rhs of SDE NLS via the SRKII.

    [DA VERIFICARE LE FORMULE]

    References
    ----------
    Honeycutt 1992, Eq.4.11
    Quiang and Habib 2000

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
    # generate the random part
    Spsi = 0
    Sphi = 0
    # build the equations
    Fpsi, Fphi = SDENLS_eq_fast(psi, phi)
    psi1 = psi + dz * Fpsi + npsi * psi * HEUN_rand()
    phi1 = phi + dz * Fphi + nphi * phi * HEUN_rand()
    Gpsi, Gphi = SDENLS_eq_fast(psi1, phi1)
    return (
        psi + 0.5 * dz * (Fpsi + Gpsi) + npsi * psi * HEUN_rand(),
        phi + 0.5 * dz * (Fphi + Gphi) + nphi * phi * HEUN_rand(),
    )


def coordinates(input):
    """Return the coordinate x with a given input."""
    xmin = input["xmin"]
    xmax = input["xmax"]
    nx = input["nx"]
    x = np.linspace(xmin, xmax, num=nx, dtype=np.double)
    dx = x[1] - x[0]
    return x, dx


def evolve_SDE_NLS_Heun(input):
    """Evolve according to the NLS with Heun algol.

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

    Returns
    -------
    A dictionary out with various output

    """
    # TODO: detail the output and input in comments

    # TODO: introdurre un dict di default per i parametri
    # in input and also for the out

    start_time = time.time()
    global minus_kx_square, dz, npsi, nphi, nx, chi, twochi, cxx

    # Extract parameter out of the dictionary
    zmax = input["zmax"]
    nx = input["nx"]
    nz = input["nz"]
    nplot = input["nplot"]
    cxx = input["cxx"]
    chi = input["chi"]
    n0 = input["n0"]
    plot_level = input["plot_level"]
    verbose_level = input["verbose_level"]

    # additional variables
    twochi = 2.0 * chi

    # coordinates
    x, dx = coordinates(input)

    # initial condition
    psi0 = input["psi0"]
    phi0 = input["phi0"]

    #  wavenumbers
    [kx, minus_kx_square] = wavenumbers(x)

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot

    # noise coefficients
    if input["noise"]:
        npsi = np.sqrt(1j * 2.0 * chi / n0) * np.sqrt(3.0 * dz / dx)
        nphi = np.sqrt(-1j * 2.0 * chi / n0) * np.sqrt(3.0 * dz / dx)
    else:
        npsi = 0.0
        nphi = 0.0

    # vector of longitudinal points
    z = 0.0
    zplot = np.zeros((nplot + 1,), dtype=np.double)
    zplot[0] = z

    #  store 2D matrices
    psi2D = np.zeros((nx, nplot + 1), dtype=np.complex64)
    psi2D[:, 0] = psi0
    phi2D = np.zeros((nx, nplot + 1), dtype=np.complex64)
    phi2D[:, 0] = psi0

    # store observable quantities
    powers = np.zeros(zplot.shape, dtype=np.double)
    mean_xs = np.zeros(zplot.shape, dtype=np.double)
    mean_square_xs = np.zeros(zplot.shape, dtype=np.double)

    # initial values for the observables
    powers[0], mean_xs[0], mean_square_xs[0] = moments(x, psi0, phi0)

    # open figure
    if plot_level > 0:
        plt.figure(1)
    # main loop
    psi = psi0
    phi = phi0
    for iplot in range(nplot):
        for iz in range(nz):
            psi, phi = HEUN_step(psi, phi)
            z = z + dz
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot " + repr(iplot + 1) + " of " + repr(nplot))
        if plot_level > 0:
            plt.figure(1)
            plt.plot(x, np.abs(psi0), "k")
            plt.plot(x, np.abs(psi))
            plt.title(repr(iplot) + " z=" + repr(z))
            plt.xlabel("x")
            plt.show()
        # store
        psi2D[:, iplot + 1] = psi
        phi2D[:, iplot + 1] = phi
        zplot[iplot + 1] = z
        powers[iplot + 1], mean_xs[iplot + 1], mean_square_xs[iplot + 1] = moments(
            x, psi, phi
        )

    # timing
    end_time = time.time()
    total_time = end_time - start_time

    # store output
    out = dict()
    out["input"] = input
    out["zplot"] = zplot
    out["powers"] = powers
    out["psi2D"] = psi2D
    out["phi2D"] = phi2D
    out["mean_xs"] = mean_xs
    out["mean_square_xs"] = mean_square_xs
    out["time_seconds"] = total_time

    if verbose_level > 0:
        print("Run time (seconds) " + repr(total_time))

    # Return
    return out


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
    global minus_kx_square, dz, dzsquare, npsi, nphi, halfnpsisquare, halfnphisquare, nx, chi, twochi, cxx, sqrt12dz

    # Extract parameter out of the dictionary
    zmax = input["zmax"]
    nx = input["nx"]
    nz = input["nz"]
    nplot = input["nplot"]
    cxx = input["cxx"]
    chi = input["chi"]
    n0 = input["n0"]
    plot_level = input["plot_level"]
    verbose_level = input["verbose_level"]
    make_step = input["step"]

    # additional variables
    twochi = 2.0 * chi

    # coordinates
    x, dx = coordinates(input)

    # initial condition
    psi0 = input["psi0"]
    phi0 = input["phi0"]

    #  wavenumbers
    [kx, minus_kx_square] = wavenumbers(x)

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot
    dzsquare = dz**2  # used by Milstein algol

    # Set the scale for the noise depending on the chosen step algo
    if make_step == HEUN_step:
        # scale for HEUN algol noise
        noise_scale = np.sqrt(3.0 * dz / dx)
        if verbose_level > 1:
            print("HEUN algol chosen")
    elif make_step == EULER_step:
        # scale for EULER algol noise
        noise_scale = np.sqrt(dz / dx)
        if verbose_level > 1:
            print("EULER algol chosen")
    elif make_step == MILSTEIN_step:
        # scale for MILSTEIN algol noise
        noise_scale = np.sqrt(dz / dx)
        if verbose_level > 1:
            print("MILSTEIN algol chosen")
    else:
        print("ERROR no make step function specified")
        return

    # noise coefficients # TODO check noise coefficients!
    if input["noise"]:
        chinoise = input["chi"]
        if "chinoise" in input:  # check if a chinoise is defined
            chinoise = input["chinoise"]
        npsi = np.sqrt(2.0 * 1j * chinoise / n0) * noise_scale
        nphi = np.sqrt(-2.0 * 1j * chinoise / n0) * noise_scale
        halfnpsisquare = 0.5 * (npsi**2)  # used by Milstein algols
        halfnphisquare = 0.5 * (nphi**2)  # used by Milstein algols
    else:
        npsi = 0.0
        nphi = 0.0
        halfnpsisquare = 0.0
        halfnphisquare = 0.0

    # vector of longitudinal points
    z = 0.0
    zplot = np.zeros((nplot + 1,), dtype=np.float64)
    zplot[0] = z

    #  store 2D matrices
    psi2D = np.zeros((nx, nplot + 1), dtype=np.complex64)
    psi2D[:, 0] = psi0
    phi2D = np.zeros_like(psi2D)
    phi2D[:, 0] = phi0

    # store observable quantities
    powers = np.zeros(zplot.shape, dtype=np.float64)
    mean_xs = np.zeros_like(powers)
    mean_square_xs = np.zeros_like(mean_xs)

    # initial values for the observables
    powers[0], mean_xs[0], mean_square_xs[0] = moments(x, psi0, phi0)

    # open figure
    if plot_level > 1:
        figura = plt.figure(1)
        ax = figura.add_subplot(111)
    # main loop
    psi = psi0
    phi = phi0
    for iplot in range(nplot):
        for iz in range(nz):
            psi, phi = make_step(psi, phi)
            z = z + dz
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot " + repr(iplot + 1) + " of " + repr(nplot))
        if plot_level > 1:
            plt.figure(1)
            plt.plot(x, np.abs(psi0), "k+")
            plt.plot(x, np.abs(psi))
            plt.title(repr(iplot) + " z=" + repr(z))
            plt.xlabel("x")
            figura.canvas.draw()
            figura.canvas.flush_events()
        #            plt.show()
        # store
        psi2D[:, iplot + 1] = psi
        phi2D[:, iplot + 1] = phi
        zplot[iplot + 1] = z
        powers[iplot + 1], mean_xs[iplot + 1], mean_square_xs[iplot + 1] = moments(
            x, psi, phi
        )

    # timing
    end_time = time.time()
    total_time = end_time - start_time

    # store output
    out = dict()
    out["input"] = input
    out["zplot"] = zplot
    out["powers"] = powers
    out["psi2D"] = psi2D
    out["phi2D"] = phi2D
    out["mean_xs"] = mean_xs
    out["mean_square_xs"] = mean_square_xs
    out["time_seconds"] = total_time

    if verbose_level > 0:
        print("Run time (seconds) %6.2f " % (total_time))

    # Return
    return out


def evolve_NLS(input):
    """Evolve deterministic NLS with split-step algol.


    +1j psi_t + c_xx psi_xx + 2.0 chi psi^2 conjugate(psi) = 0

    psi_t = 1j*c_xx*psi_xx+1j*2.0*chi*psi^2*conj(psi)


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
    plot_level=1

    plot_level = 2 plot intermediate fields

    Returns
    -------
    A dictionary out with various output

    """
    # TODO: detail the output and input in comments

    # TODO: introdurre un dict di default (con config file) per i parametri
    # in input and also for the out

    start_time = time.time()
    global twochi, twochi1j, conv

    # Extract parameter out of the dictionary
    zmax = input["zmax"]
    nx = input["nx"]
    nz = input["nz"]
    nplot = input["nplot"]
    cxx = input["cxx"]
    chi = input["chi"]
    plot_level = input["plot_level"]
    verbose_level = input["verbose_level"]

    # additional variables
    twochi = 2.0 * chi
    twochi1j = twochi * 1j

    # coordinates
    x, dx = coordinates(input)

    # initial condition
    psi0 = input["psi0"]

    #  wavenumbers
    [kx, minus_kx_square] = wavenumbers(x)

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot

    print("DEBUG dz=" + repr(dz) + " " + " 1j=" + repr(1j))
    # vector of longitudinal points
    z = 0.0
    zplot = np.zeros((nplot + 1,), dtype=dtreal)
    zplot[0] = z

    # store 1D vectors
    psi = np.zeros_like(psi0)
    I1 = np.zeros_like(psi0)

    #  store 2D matrices
    psi2D = np.zeros((nx, nplot + 1), dtype=dtcomplex)
    psi2D[:, 0] = psi0

    # store observable quantities
    powers = np.zeros(zplot.shape, dtype=np.float64)
    mean_xs = np.zeros_like(powers)
    mean_square_xs = np.zeros_like(mean_xs)

    # transfer function for the convolution
    conv = np.zeros(psi0.shape, dtype=dtcomplex)
    conv = 1j * cxx * minus_kx_square * 0.5 * dz
    conv = np.exp(conv)

    # initial values for the observables
    powers[0], mean_xs[0], mean_square_xs[0] = moments(x, psi0, np.conjugate(psi0))

    # open figure
    if plot_level > 1:
        figura = plt.figure(1)
        ax = figura.add_subplot(111)
    # main loop
    psi = psi0
    for iplot in range(nplot):
        for iz in range(nz):
            # half dispersive step
            psi = fft.ifft(fft.fft(psi) * conv)
            # full nonlinear step
            I1 = np.abs(psi) ** 2
            psi = psi * np.exp(twochi1j * I1 * dz)
            # half dispersive
            psi = fft.ifft(fft.fft(psi) * conv)
            # psi = NLS_step(psi)
            z = z + dz
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot " + repr(iplot + 1) + " of " + repr(nplot))
        if plot_level > 1:
            plt.figure(1)
            plt.plot(x, np.abs(psi0), "k+")
            plt.plot(x, np.abs(psi))
            plt.title(repr(iplot) + " z=" + repr(z))
            plt.xlabel("x")
            figura.canvas.draw()
            figura.canvas.flush_events()
        #            plt.show()
        # store
        psi2D[:, iplot + 1] = psi
        zplot[iplot + 1] = z
        powers[iplot + 1], mean_xs[iplot + 1], mean_square_xs[iplot + 1] = moments(
            x, psi, np.conjugate(psi)
        )

    # timing
    end_time = time.time()
    total_time = end_time - start_time

    # store output
    out = dict()
    out["input"] = input
    out["zplot"] = zplot
    out["powers"] = powers
    out["psi2D"] = psi2D
    out["mean_xs"] = mean_xs
    out["mean_square_xs"] = mean_square_xs
    out["time_seconds"] = total_time

    if verbose_level > 0:
        print("Run time (seconds) %6.2f " % (total_time))

    # Return
    return out
