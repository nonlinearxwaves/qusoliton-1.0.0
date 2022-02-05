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

# %%
import time
import numpy as np
import cupy as cp
import cupy.fft as fft
import matplotlib.pyplot as plt
from hurry.filesize import size, si

# %% datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype

# datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype

# %% return the wave numbers


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
    intensity = cp.real(psi*phi)
    return cp.sum(cp.sum(intensity))*dxdy


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
    return psi/cp.sqrt(P), phi/cp.sqrt(P), P


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
    i_norm = cp.real(psi_norm*phi_norm)
    mean_x = cp.sum(X*i_norm)*dxdy
    mean_y = cp.sum(Y*i_norm)*dxdy
    mean_x2 = cp.sum((X**2)*i_norm)*dxdy
    mean_y2 = cp.sum((Y**2)*i_norm)*dxdy
    mean_r2 = mean_x2+mean_y2
    return P, mean_x, mean_y, mean_x2, mean_y2, mean_r2

# %% unnormalized participation ratio


def Q(f, g, dkxdky=1.0):
    """ Return the unnormalized participation ratio

    Is to be divided by the square energy 

    """
    return dkxdky*cp.sum(cp.sum(
        cp.real(
            cp.multiply(cp.square(f), cp.square(g))
        )
    ))
# %% spectral intensity


def spectral_intensity(psi, phi):
    return cp.real(cp.fft.fft2(psi)*cp.fft.fft2(phi))


# %% costant for random number generation
ONE_SIX = 0.166666666666666665
FIVE_SIX = 0.8333333333333334


# %% return the coordinates as linear vector
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

# %% return coordinates and spectral coordinates as 2D matrices


def meshgrid_coords(input):
    x, y, _, _ = coordinates(input)
    X, Y = np.meshgrid(x, y)
    kx, ky, _, _ = wavenumbers(x, y)
    KX, KY = np.meshgrid(kx, ky)
    return X, Y, KX, KY

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
def MILSTEIN_step():
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
    global psi, phi, psi_xx_yy, phi_xx_yy, Fpsi, Fphi, tmp, LAPL
    global halfnphisquare, halfnpsisquare, npsi, nphi
    global z, dz, chi, nx, ny, dzsquare

    psi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(psi))
    phi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(phi))
    tmp = psi*phi
    Fpsi = +1j*psi_xx_yy+1j*2.0*chi*psi*tmp
    Fphi = -1j*phi_xx_yy-1j*2.0*chi*phi*tmp
    DW1 = cp.random.normal(size=(ny, nx))
    DW2 = cp.random.normal(size=(ny, nx))
    DWpsi = npsi*DW1
    DWphi = nphi*DW2
    DWpsi += halfnpsisquare*(DW1**2-dzsquare)
    DWphi += halfnphisquare*(DW2**2-dzsquare)
    psi = psi + dz*Fpsi + psi*DWpsi
    phi = phi + dz*Fphi + phi*DWphi
    z = z+dz


# %% define make_step function for MILSTEIN step with potential and absorber
def MILSTEIN_potential_step():
    """Make a step with MILSTEIN algol with potential and absorber



    """
    global psi, phi, psi_xx_yy, phi_xx_yy, Fpsi, Fphi, tmp, LAPL
    global pot, absorb
    global halfnphisquare, halfnpsisquare, npsi, nphi
    global z, dz, chi, nx, ny, dzsquare

    psi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(psi))
    phi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(phi))
    tmp = psi*phi
    Fpsi = +1j*psi_xx_yy+1j*2.0*chi*psi*tmp-1j*pot*psi-absorb*psi
    Fphi = -1j*phi_xx_yy-1j*2.0*chi*phi*tmp+1j*pot*phi-absorb*phi
    DW1 = cp.random.normal(size=(ny, nx))
    DW2 = cp.random.normal(size=(ny, nx))
    DWpsi = npsi*DW1
    DWphi = nphi*DW2
    DWpsi += halfnpsisquare*(DW1**2-dzsquare)
    DWphi += halfnphisquare*(DW2**2-dzsquare)
    psi = psi + dz*Fpsi + psi*DWpsi
    phi = phi + dz*Fphi + phi*DWphi
    z += dz

# %% make_step MILSTEIN step with time dependent potential and absorber


def MILSTEIN_time_potential_step():
    """Make a step with MILSTEIN algol with potential and absorber
    Time dependent case

    Scales down the potential up to end of the simulation

    """
    global psi, phi, psi_xx_yy, phi_xx_yy, Fpsi, Fphi, tmp, LAPL
    global pot, absorb
    global halfnphisquare, halfnpsisquare, npsi, nphi
    global z, dz, chi, nx, ny, dzsquare, izmax

    psi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(psi))
    phi_xx_yy = cp.fft.ifft2(LAPL*cp.fft.fft2(phi))
    tmp = psi*phi
    Fpsi = +1j*psi_xx_yy+1j*2.0*chi*psi*tmp-1j*(1-z*izmax)*pot*psi-absorb*psi
    Fphi = -1j*phi_xx_yy-1j*2.0*chi*phi*tmp+1j*(1-z*izmax)*pot*phi-absorb*phi
    DW1 = cp.random.normal(size=(ny, nx))
    DW2 = cp.random.normal(size=(ny, nx))
    DWpsi = npsi*DW1
    DWphi = nphi*DW2
    DWpsi += halfnpsisquare*(DW1**2-dzsquare)
    DWphi += halfnphisquare*(DW2**2-dzsquare)
    psi = psi + dz*Fpsi + psi*DWpsi
    phi = phi + dz*Fphi + phi*DWphi
    z += dz


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
    A dictionary out with various output, some of them as cp other as np 

    intens   // cp matrix
    kintens  // np matrix


    """
    # TODO: detail the output and input in comments

    # TODO: introdurre un dict di default per i parametri
    # in input and also for the out

    # %% free cuda memory
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # print used memory
    print("Total CUDA memory " + repr(size(mempool.total_bytes())))

    # %% define global variables
    global psi, phi, psi_xx_yy, phi_xx_yy, Fpsi, Fphi, tmp, LAPL
    global pot, absorb
    global halfnphisquare, halfnpsisquare, npsi, nphi
    global z, dz, chi, nx, ny, dzsquare, izmax

    # %%
    start_time = time.time()

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
    set_potential = False  # flag for using algol with potential

    # coordinates
    xmin = input['xmin']
    xmax = input['xmax']
    nx = input['nx']
    x = cp.linspace(xmin, xmax, num=nx, dtype=dtreal)
    dx = x[1]-x[0]
    ymin = input['ymin']
    ymax = input['ymax']
    ny = input['ny']
    y = cp.linspace(ymin, ymax, num=ny, dtype=dtreal)
    dy = y[1]-y[0]

    # coordinate points
    X, Y = cp.meshgrid(x, y)
    Xnp = cp.asnumpy(X)
    Ynp = cp.asnumpy(Y)

    # initial condition
    psi0np = input['psi0']
    phi0np = input['phi0']

    # potential and absorber
    if 'potential' in input:
        potnp = input['potential']
    if 'absorber' in input:
        absorbernp = input['absorber']

    #  wavenumbers
    nx = x.shape[0]
    dx = x[1]-x[0]
    kx = cp.zeros(x.shape, dtype=dtreal)
    kx = 2.0*np.pi*cp.fft.fftfreq(nx, d=dx)
    ny = y.shape[0]
    dy = y[1]-y[0]
    ky = cp.zeros(y.shape, dtype=dtreal)
    ky = 2.0*np.pi*cp.fft.fftfreq(ny, d=dy)

    # area element in the momentum space
    dkx = kx[1]-kx[0]
    dky = ky[1]-ky[0]
    dkxdky = dkx*dky

    # spectral vectors
    KXnp, KYnp = np.meshgrid(np.sort(cp.asnumpy(kx)), np.sort(cp.asnumpy(ky)))
    MKX2, MKY2 = cp.meshgrid(-cp.square(kx), -cp.square(ky))

    # laplacian
    LAPL = cxx*MKX2+cyy*MKY2

    # longitudinal step (CHECK HERE)
    dz = (zmax / nz) / nplot
    dzsquare = dz**2  # used by Milstein algol
    izmax = 1.0/zmax

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
    elif make_step == MILSTEIN_potential_step:
        # scale for MILSTEIN algol noise
        noise_scale = np.sqrt(dz/(dx*dy))
        if verbose_level > 1:
            print("MILSTEIN with potential algol chosen")
        set_potential = True
    elif make_step == MILSTEIN_time_potential_step:
        # scale for MILSTEIN algol noise
        noise_scale = np.sqrt(dz/(dx*dy))
        if verbose_level > 1:
            print("MILSTEIN with potential algol chosen")
        set_potential = True
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

    # store observable quantities
    peaks = cp.zeros(zplot.shape, dtype=np.double)
    powers = cp.zeros(zplot.shape, dtype=np.double)
    kpowers = cp.zeros(zplot.shape, dtype=np.double)
    Qs = cp.zeros(zplot.shape, dtype=np.double)
    mean_xs = cp.zeros(zplot.shape, dtype=np.double)
    mean_ys = cp.zeros(zplot.shape, dtype=np.double)
    mean_square_xs = cp.zeros(zplot.shape, dtype=np.double)
    mean_square_ys = cp.zeros(zplot.shape, dtype=np.double)
    mean_square_rs = cp.zeros(zplot.shape, dtype=np.double)

    # initial values for the observables
    psi0cp = cp.asarray(psi0np)
    phi0cp = cp.asarray(phi0np)

    # intensity and spectral intensity
    intens = cp.zeros((ny, nx, nplot+1), dtype=np.double)
    kintens = cp.zeros((ny, nx, nplot+1), dtype=np.double)

    # potential and absorber
    pot = cp.zeros((ny, nx), dtype=np.double)
    absorb = cp.zeros((ny, nx), dtype=np.double)
    if set_potential:
        pot = cp.asarray(potnp)
        absorb = cp.asarray(absorbernp)

    # print used memory
    print("Used CUDA memory " + repr(size(mempool.used_bytes())))

    # initial values for the observables
    peaks[0] = peak(psi0cp, phi0cp)
    powers[0], mean_xs[0], mean_ys[0], \
        mean_square_xs[0], mean_square_ys[0], \
        mean_square_rs[0] = moments(X, Y, psi0cp, phi0cp)
    kpowers[0] = power(cp.fft.fft2(psi0cp),
                       cp.fft.fft2(phi0cp), dkxdky)
    Qs[0] = Q(cp.fft.fft2(psi0cp), cp.fft.fft2(phi0cp), dkxdky)

    # store initial intensity and spectral intensity
    intens[:, :, 0] = cp.real(cp.multiply(psi0cp, phi0cp))
    kintens[:, :, 0] = cp.real(
        cp.multiply(cp.fft.fft2(psi0cp), cp.fft.fft2(phi0cp)))

    # open figure
    if plot_level > 1:
        fig = plt.figure(figsize=plt.figaspect(0.5))

    # main loop
    psi = cp.asarray(psi0cp)
    phi = cp.asarray(phi0cp)
    fpsi = cp.asarray(psi0cp)
    fphi = cp.asarray(phi0cp)
    tmp = cp.zeros_like(psi0cp)
    Fpsi = cp.zeros_like(psi)
    Fphi = cp.zeros_like(psi)
    psi_xx_yy = cp.zeros_like(psi)
    phi_xx_yy = cp.zeros_like(psi)
    for iplot in range(1, nplot+1):  # remark start form 1
        for iz in range(nz):
            make_step()
        # temporary current field solution and initial one
        if verbose_level > 1:
            print("Current plot %d of %d, power = %5.2f "
                  % (iplot+1, nplot, np.asscalar(cp.asnumpy(powers[iplot]))))
        if plot_level > 1:
            tmpplot1 = cp.asnumpy(intens[:, :, iplot-1])
            tmpplot2 = np.fft.fftshift(cp.asnumpy(kintens[:, :, iplot-1]))
            fig = plt.figure(figsize=plt.figaspect(0.5))
            axa = fig.add_subplot(1, 2, 1, projection='3d')
            axa.set_title("density")
            axb = fig.add_subplot(1, 2, 2, projection='3d')
            axb.set_title("spectral density")
            surf_intens = axa.plot_surface(Xnp, Ynp, tmpplot1)
            surf_kintens = axb.plot_surface(KXnp, KYnp, tmpplot2)
            plt.show()
        # store
        fpsi = cp.fft.fft2(psi)
        fphi = cp.fft.fft2(phi)
        zplot[iplot] = z
        powers[iplot], \
            mean_xs[iplot], \
            mean_ys[iplot], \
            mean_square_xs[iplot], \
            mean_square_ys[iplot], \
            mean_square_rs[iplot] = moments(X, Y, psi, phi)
        peaks[iplot] = peak(psi, phi)
        kpowers[iplot] = power(
            fpsi,
            fphi, dkxdky)
        Qs[iplot] = Q(fpsi, fphi, dkxdky)

        # store initial intensity and spectral intensity
        intens[:, :, iplot] = cp.real(cp.multiply(psi, phi))
        kintens[:, :, iplot] = cp.real(
            cp.multiply(cp.fft.fft2(psi), cp.fft.fft2(phi)))

    # timing
    end_time = time.time()
    total_time = end_time-start_time

    # store output (improve the output for 3D or similar)
    out = dict()
    out['input'] = input
    out['X'] = Xnp
    out['Y'] = Ynp
    out['KX'] = KXnp
    out['KY'] = KYnp
    out['zplot'] = zplot
    out['powers'] = powers
    out['kpowers'] = kpowers
    out['Qs'] = Qs
    out['intens'] = intens
    out['kintens'] = kintens
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
