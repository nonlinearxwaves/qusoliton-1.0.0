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
from termcolor import colored
import pdb
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import time
# -*- coding: utf-8 -*-
""" Module for NLS stochastic in 3D

Module with routines for the integration of the
Stochastic NLS

-------------------------------
Deterministic version of the equation

1j psi_t + c_xx psi_xx + c_yy psi_yy + c_zz psi_zz + 2 chi |psi|^2 psi =0

psi_t = 1j c_xx psi_xx+ 1j c_yy psi_yy + c_zz psi_zz + 1j 2 chi |psi|^2 psi


-------------------------------
Stochastic version of the equation


psi_t = 1j c_xx psi_xx + 1j c_yy psi_yy + 1j c_zz psi_zz  + 1j 2 chi psi^2 phi + noise_psi phi xi

phi_t = -1j c_xx phi_xx-1j c_yy phi_yy + 1j c_zz psi_zz - 1j 2 chi psi phi^2 + noise_phi phi nu

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

TODO: fix this documentation

@author: claudio
@version:9 march 2020
"""

# TODO: optimize 1j*cxx as 1jcxx


# datatypes
dtreal = np.double  # real datatype
dtcomplex = np.complex64  # complex datatype
dtreal = np.float16  # real datatype
dtcomplex = np.complex64  # complex datatype

# global variables
nx = 2
ny = 2
nt = 2
cxx = 1.0
cyy = 1.0
ctt = 1.0
chi = 1.0
twochi = 2.0
dz = 0.1
zmax = 1.0
dzsquare = 0.01
npsi = 0.0
nphi = 0.0
halfnpsisquare = 0.0
halfnphisquare = 0.0
make_step = None
# noise coefficient
nshots = 1
shot_number = 0
nphoton = 10
alphanoise = 0
# output levels
plot_level = 3
verbose_level = 3
save_all_shots = False
filerootshot = '.tests/data/datashot'
# current status
psi = 0
phi = 0
# initial conditions
psi0 = 0
phi0 = 0
# 4D storage
psi3D = 0
phi3D = 0
# other global variables
X = None
Y = None
T = None
nz = 2
nplot = 2
zplot = None
xmin = None
xmax = None
tmin = None
tmax = None
ymin = None
ymax = None
x = None
y = None
t = None
z = None
dx = None
dy = None
dt = None
dV = None
kx = None
ky = None
kt = None
KX = None
KY = None
KT = None
Volume = 0.0
# nonlfglocality parameters
sigma_local = 1
sigma_nonlocal = 1
# iterations parameters
iter_implicit = 3
# relaxation parameters
relax_eig_tol = 1.0e-6
relax_tol = 1.0e-8
relax_iter_linear = 100
relax_alpha = 0.0001
relax_niter = 100
# global special variables
__conv__ = None
__conjconv__ = None
__lapl__ = None  # laplacian operator
nonlocal_kernel = None
__sqrt_nonlocal_kernel__ = None
__stratonovich__ = 1j  # stratovich correction
# costant for random number generation
__ONE_SIX__ = 0.166666666666666665
__FIVE_SIX__ = 0.833333333
__SQRTPI__ = dtcomplex(np.sqrt(1j, dtype=dtcomplex))
__SQRTMI__ = dtcomplex(np.sqrt(-1j, dtype=dtcomplex))
# convergece
__max_error__ = 0.0
# observable quantities
peaks = 0
powers = 0
mean_pos = 0
mean_square_pos = 0


def wavenumbers(x, y, t):
    """Return the wavenumbers for fft corresponding to x

    Parameters
    ----------------------
    x: array of doubles, must have shape (nx,)
    y: array of doubles, must have shape (ny,)
    t: array of doubles, must have shape (nt,)

    Returns
    ----------------------
    kx  : vector of wavenumbers, shape (nx, )
    minus_kx_square : -kx**2 for second derivative, shape (nx, )
    ky  : vector of wavenumbers, shape (ny, )
    minus_ky_square : -ky**2 for second derivative, shape (ny, )
    kt  : vector of wavenumbers, shape (nt, )
    minus_kt_square : -kt**2 for second derivative, shape (nt, )

    """
    kx = np.zeros(x.shape, dtype=dtreal)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    ky = np.zeros(y.shape, dtype=dtreal)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    kt = np.zeros(t.shape, dtype=dtreal)
    kt = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    return kx, ky, kt, -(kx ** 2), -(ky ** 2), -(kt ** 2)


def peak():
    """Return peak intensity of a field as the max of real(psi*phi)

    Parameters
    --------------
    psi : array of complex
    phi : array of complex

    Returns
    --------------
    The maximum of real(psi*phi)

    """

    intensity = np.real(psi * phi)
    return np.amax(intensity)


def power():
    """Return power of a field as the integral of real(psi*phi)

    Parameters
    --------------
    psi : array of complex
    dx : increment along x
    dy : increment along y

    Returns
    --------------
    The integral of |psi|**2 , if the field is normalized this is 1.0

    """
    intensity = np.real(psi * phi)
    return np.sum(intensity) * dV


def normalize():
    """Return a field normalized to have unitary power as phi*psi

    Parameters
    --------------
    x : array of reals
    psi : array of complex

    Returns
    --------------
    normalize field psi
    power of the input field

    """
    P = power()
    return psi / np.sqrt(P), phi / np.sqrt(P), P


def moments():
    """Return first moments wrt x of a field (psi, phi)

    Parameters
    --------------
    X : 2D array (meshgrid) of reals
    Y : 2D array (meshgrid) of reals
    psi : array of complex

    Returns
    --------------
    power : power of the beam
    mean_x : mean of x wrt |psi|^2
    mean_y : mean of y wrt |psi|^2
    mean_z : mean of z wrt |psi|^2
    mean_x2 : mean of x**2 wrt |psi|^2
    mean_y2 : mean of x**2 wrt |psi|^2
    mean_z2 : mean of z**2 wrt |psi|^2
    mean_r2 : mean of x**2 + y**2 + z**2 wrt |psi|^2

    """
    psi_norm, phi_norm, P = normalize()
    i_norm = np.real(psi_norm * phi_norm)
    mean_x = np.sum(X * i_norm) * dV
    mean_y = np.sum(Y * i_norm) * dV
    mean_t = np.sum(T * i_norm) * dV
    mean_x2 = np.sum((X ** 2) * i_norm) * dV
    mean_y2 = np.sum((Y ** 2) * i_norm) * dV
    mean_t2 = np.sum((T ** 2) * i_norm) * dV
    mean_r2 = mean_x2 + mean_y2 + mean_t2
    return P, mean_x, mean_y, mean_t, mean_x2, mean_y2, mean_t2, mean_r2


# %%
def coordinates():
    """Return the coordinate x, y with a given input"""
    x = np.linspace(xmin, xmax, num=nx, dtype=dtreal)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, num=ny, dtype=dtreal)
    dy = y[1] - y[0]
    t = np.linspace(tmin, tmax, num=nt, dtype=dtreal)
    dt = t[1] - t[0]
    return x, y, t, dx, dy, dt


# %% laplacian function
def d_xx_yy_tt(psi):
    """Return the second derivative of the field psi by fft

    Parameters
    --------------
    psi : array of complex64 for the field

    Returns
    --------------
    cxx psi_xx+ cyy psi_yy + ctt psi_tt : second derivatives with respect to x

    """
    # this function is to remove
    global LAPL
    return fft.ifft2(LAPL * fft.fft2(psi))


#    return fft.ifft2(LAPL*fft.fft2(psi))/(nx*ny)

# %% main equations
def SDENLS_eq(psi, phi):
    """Return the deterministic rhs of SDE NLS

    Parameters
    --------------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    Returns
    --------------
    Two output complex vectors

    +1j c_xx psi_xx+c_yy psi_yy+c_zz psi_zz + 1j 2.0 chi psi^2 phi
    -1j c_xx phi_xx+c_yy phi_yy+c_z phi_zz - 1j 2.0 chi psi phi^2


    """
    global chi
    psi_xx_yy_tt = d_xx_yy_tt(psi)
    phi_xx_yy_tt = d_xx_yy_tt(phi)
    tmp = psi * phi
    psi_ = +1j * psi_xx_yy_tt + 1j * 2.0 * chi * psi * tmp
    phi_ = -1j * phi_xx_yy_tt - 1j * 2.0 * chi * phi * tmp
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

    +1j psi_t + psi_nabla + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + psi_nabla - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """
    global dz, halfdz, npsi, nphi
    Fpsi, Fphi = SDENLS_eq(psi, phi)
    psi1 = psi + dz * Fpsi + npsi * psi * HEUN_rand()
    phi1 = phi + dz * Fphi + nphi * phi * HEUN_rand()
    Gpsi, Gphi = SDENLS_eq(psi1, phi1)
    psi_ = psi + halfdz * (Fpsi + Gpsi) + npsi * psi * HEUN_rand()
    phi_ = phi + halfdz * (Fphi + Gphi) + nphi * phi * HEUN_rand()
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
    psi1 = psi + dz * Fpsi + npsi * psi * np.random.normal(size=(ny, nx))
    phi1 = phi + dz * Fphi + nphi * phi * np.random.normal(size=(ny, nx))
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
    DWpsi = npsi * DW1
    DWphi = nphi * DW2
    DWpsi += halfnpsisquare * (DW1 ** 2 - dzsquare)
    DWphi += halfnphisquare * (DW2 ** 2 - dzsquare)
    psi1 = psi + dz * Fpsi + psi * DWpsi
    phi1 = phi + dz * Fphi + phi * DWphi
    return psi1, phi1


def linearhalfstep(psi, phi):
    """ make a linear half step for the Drummond algol
    """

    return fft.ifftn(__conv__*fft.fftn(psi)),\
        fft.ifftn(__conjconv__*fft.fftn(phi))


def nonlocal_noise():
    """ generate nonlocal_noise and its Fourier transform """

    FDWpsi = np.random.normal(0.0, 1.0, (nx, ny, nt))
    FDWphi = np.random.normal(0.0, 1.0, (nx, ny, nt))

    FDWpsi = __sqrt_nonlocal_kernel__*FDWpsi
    FDWphi = __sqrt_nonlocal_kernel__*FDWphi

    DWpsi = np.fft.ifftn(FDWpsi)
    DWphi = np.fft.ifftn(FDWphi)

#    pdb.set_trace()
    return DWpsi, DWphi


# %% main evolution algol with Drummond iterative scheme


def DRUMMOND_step(psi, phi):
    """Make a step with DRUMMOND iterative algol

    Parameters
    ----------
    psi : array of complex64 for the field psi
    phi : array of complex64 for the field phi

    IMPORTANT, for reason of speed the noise coefficients are defined as
    npsi = sqrt(+2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    nphi = sqrt(-2.0*1j*chi/n0)*sqrt(3.0*dz/(dx*dy))
    and must be calculate befor calling this function

    Returns
    -------
    Two next update of the equations with the model

    +1j psi_t + c_xx psi_xx + 1j 2.0 chi psi^2 phi + npsi psi rand_psi
    -1j phi_t + c_xx psi_xx - 1j 2.0 chi psi phi^2 + nphi phi rand_phi


    """

    global __max_error__

    # make a linear half step
    psi, phi = linearhalfstep(psi, phi)

    # compute noise
    DWpsi, DWphi = nonlocal_noise()

    # check if NaN

    # store the current status
    df = np.copy(psi)
    dg = np.copy(phi)
    f0 = np.copy(psi)
    g0 = np.copy(phi)

    # allocate temporary arrays
    tmpf1 = np.copy(df)  # check this may use too much memory
    tmpg1 = np.copy(dg)
    nonlocalfg = np.copy(df)

    # iterations
    __max_error__ = 0.0
    error_convergence = 0.0
    for iter in range(iter_implicit):  # to do should check the use of memory

        # evaluate nonlocal nonlinearity
        nonlocalfg = np.fft.ifftn(np.fft.fftn(df*dg)*nonlocal_kernel)

        tmpf1 = 2.0*1j*chi*df*nonlocalfg
        tmpg1 = -2.0*1j*chi*dg*nonlocalfg

        # add noise and stratovich correction
        df = f0+0.5*(dz*tmpf1+__SQRTMI__*df*DWpsi)-__stratonovich__*df
        dg = g0+0.5*(dz*tmpg1+__SQRTPI__*dg*DWphi)+__stratonovich__*dg

#        pdb.set_trace()
        # check convergence
        error_convergence = np.amax(np.abs(df-f0))
        if error_convergence > __max_error__:
            __max_error__ = error_convergence

    # final approximation for the field
    psi = 2.0*df-f0
    phi = 2.0*dg-g0

    # linear half step
    psi, phi = linearhalfstep(psi, phi)

    # current z (this is done outside this function)

    return psi, phi


def evaluate_nonlocal_kernel():
    """ Compute the nonlocal kernel and its sqrt """
    global nonlocal_kernel
    global __sqrt_nonlocal_kernel__
    global __stratonovich__

    nonlocal_kernel = np.zeros((nx, ny, nt), dtype=dtreal)
    __sqrt_nonlocal_kernel__ == np.zeros((nx, ny, nt), dtype=dtcomplex)

    nonlocal_kernel = 0.5/(sigma_local+sigma_nonlocal*(KX**2+KY**2+KT**2))

    noise_weight1 = alphanoise*np.sqrt(2.0*chi*dz/(dV*nphoton))

#    pdb.set_trace()
    __sqrt_nonlocal_kernel__ = noise_weight1*np.sqrt(nonlocal_kernel)

    __stratonovich__ = 2.0*np.sum(nonlocal_kernel)/(nx*ny*nt*Volume)

    __stratonovich__ = 1j*__stratonovich__*0.25*dz

    return None


def init(**kwargs):
    """ General init function

    Parameters
    ----------
    input_dic  : input dictionary


    """

    # global variables
    global x, y, t, dx, dy, dt, dV, Volume
    global nx, ny, nt
    global __conv__, __conjconv__, __lapl__
    global X, Y, T
    global kx, ky, kt
    global KX, KY, KT
    global plot_level, verbose_level
    global make_step
    global zmax, nplot, nz, dz
    global sigma_local
    global sigma_nonlocal
    global alphanoise
    global chi
    global nphoton
    global psi, phi
    global iter_implicit
    global step
    global zplot

    step = DRUMMOND_step

    assert nz > 1, "nz must be greater than 1"
    assert nplot > 1, "nplot must be greater than 1 "
    dz = (zmax / nz) / nplot
    zplot = np.linspace(0, zmax, nplot+1)

    # define the coordinates
    x, y, t, dx, dy, dt = coordinates()
    X, Y, T = np.meshgrid(x, y, t)
    dV = dx*dy*dt
    Volume = (tmax-tmin)*(xmax-xmin)*(ymax-ymin)
    # define the spectrum
    kx, ky, kt, minus_kx_square, minus_ky_square, minus_kt_square = wavenumbers(
        x, y, t)

    KX, KY, KT = np.meshgrid(kx, ky, kt)
    # compute the propagator for half step
    MKX2, MKY2, MKT2 = np.meshgrid(
        minus_kx_square, minus_ky_square, minus_kt_square)
    __conv__ = np.exp(1j*0.5*dz*(cxx*MKX2+cyy*MKY2+ctt*MKT2))
    __conjconv__ = np.exp(-1j*0.5*dz*(cxx*MKX2+cxx*MKY2+ctt*MKT2))
    __lapl__ = cxx*MKX2+cyy*MKY2+ctt*MKT2
    make_step = DRUMMOND_step

    # allocate fields if not defined
    if np.isscalar(psi):
        psi = np.zeros((nx, ny, nt), dtype=dtcomplex)
    if np.isscalar(phi):
        phi = np.zeros_like(psi)

    evaluate_nonlocal_kernel()

    return None


def initialize_observables():
    """ init the observables quantities """
    global psi3D, phi3D
    global peaks
    global powers
    global mean_pos
    global mean_square_pos
    global zplot

    #  store 4D matrices
    psi3D = np.zeros((ny, nx, nt, nplot + 1), dtype=np.complex64)
    psi3D[:, :, :, 0] = psi0
    phi3D = np.zeros((ny, nx, nt, nplot + 1), dtype=np.complex64)
    phi3D[:, :, :, 0] = phi0

    # store observable quantities
    nplot1 = nplot+1
    peaks = np.zeros((nplot1, nshots), dtype=np.double)
    powers = np.zeros((nplot1, nshots), dtype=np.double)

    # mean_pos encode x,y,t, for nplot+1 and nshots
    mean_pos = np.zeros((3, nplot1, nshots), dtype=np.double)
    # mean_square_pos encode x2,y2,t2, for nplot+1 and nshots
    mean_square_pos = np.zeros((4, nplot1, nshots), dtype=np.double)

    return None


def store_observables(iplot, ishot):
    """ store observables """
    global psi3D, phi3D
    global peaks
    global powers
    global mean_pos
    global mean_square_pos

    psi3D[:, :, :, iplot] = np.copy(psi)
    phi3D[:, :, :, iplot] = np.copy(phi)

    peaks[iplot, ishot] = peak()

    (
        powers[iplot, ishot],
        mx,
        my,
        mt,
        mx2,
        my2,
        mt2,
        mr2,
    ) = moments()

    mean_pos[0, iplot, ishot] = mx
    mean_pos[1, iplot, ishot] = my
    mean_pos[2, iplot, ishot] = mt
    mean_square_pos[0, iplot, ishot] = mx2
    mean_square_pos[1, iplot, ishot] = my2
    mean_square_pos[2, iplot, ishot] = mt2
    mean_square_pos[3, iplot, ishot] = mt2

    return None


def messages_evolution_step(iplot, ishot):
    """ display ste message during evolution """

    if verbose_level > 1:
        print(
            "Shot %d of %d, plot %d of %d,"
            " pow = %5.2f, w2 = %5.2f,"
            " max = %5.2f, z=%5.2f, err=%5.2e "
            % (ishot + 1, nshots,
               iplot + 1, nplot, powers[iplot, ishot],
               mean_square_pos[3, iplot, ishot],
               peaks[iplot, ishot], z, __max_error__)
        )

    return None


def save_shot(ishot):
    """ save data for the current shot """
    filename = filerootshot+repr(ishot)
    saveall(filename)
    return None


def evolve_SDE_NLS(input_dic):
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
    # TODO: fix this documentation
    # in input and also for the out

    start_time = time.time()
    global z
    global psi, phi  # current status
    # intialize and extract globals from the dictionary
    # init(input_dic)

    initialize_observables()

    # open figure
    if plot_level > 1:
        plt.figure(1)
    # main loop
    for shot_number in range(nshots):
        z = 0.0
        # set initial conditions
        psi = np.copy(psi0)
        phi = np.copy(phi0)
        store_observables(0, shot_number)
        for iplot in range(nplot):
            for iz in range(nz):
                psi, phi = make_step(psi, phi)
                z = z + dz
            # temporary current field solution and initial one
            messages_evolution_step(iplot, shot_number)
            # store
            store_observables(iplot+1, shot_number)
        if save_all_shots:
            save_shot(shot_number)

    # timing
    end_time = time.time()
    total_time = end_time - start_time

    # store output (improve the output for 3D or similar)
    out = dict()
    out["input"] = input
    out["time_seconds"] = total_time

    if verbose_level > 0:
        print(colored("Run time (seconds) %6.2f " % (total_time), 'green'))

    # Return
    return out


def relax_tests():
    """ make some tests for the relax function """

    global psi0, phi0

    # initial gaussian
    realprofile = np.copy(psi0)

    while (True):
        # evaluate nonlocal
        realnonlocal = np.fft.ifftn(np.fft.fftn(realprofile**2
                                                )*nonlocal_kernel).real

        fnabla = np.fft.ifftn(__lapl__*np.fft.fftn(realprofile)).real
#        plot_panels(realnonlocal)

        # test laplacian

        break

    Image.open(plot_panels(nonlocal_kernel,
                           './tests/img/currentplot.png', 'nonlocalkernel'
                           )).show()
    Image.open(plot_panels(realnonlocal,
                           './tests/img/realnonlocal.png', 'realnonlocal'
                           )).show()

    Image.open(plot_panels(fnabla,
                           './tests/img/realnonlocal.png', 'nabla'
                           )).show()

    # store final guess
    # psi0 = realprofile
    psi0 = nonlocal_kernel

    phi0 = np.copy(realprofile)

    return None


def relax():
    """ find solution by relaxation """
    global psi0, phi0

    # initial gaussian
    realprofile = np.copy(psi0)

    # nonlinear iterations
    iternl = 0
    while (True):
        # evaluate nonlocal
        realnonlocal = np.fft.ifftn(np.fft.fftn(realprofile**2
                                                )*nonlocal_kernel).real

        # linear iterations
        iterl = 0
        eigenvalue = 1.0
        realprofile_old = np.copy(realprofile)
        while (True):
            # linear iteration with constant potential
            eigenvalue_old = eigenvalue

            fnabla = np.fft.ifftn(__lapl__*np.fft.fftn(realprofile)).real

            nl = -fnabla-2.0*chi*realnonlocal*realprofile

            realprofile = realprofile-relax_alpha*nl

            norma = np.sum(realprofile**2)*dV

            eigenvalue = np.sum(nl*realprofile)*dV/norma

            deig = np.abs(1.0-eigenvalue/eigenvalue_old)

            if (iterl > 0) & (verbose_level > 2):
                print(
                    " Linear e, deltae, iter, tot, %5.2e, %5.2e, %d, %d "
                    % (eigenvalue, deig, iterl, relax_iter_linear))

            # find the maximum
            lmaxfr = np.max(realprofile)
            # normalize to keep the maximum constant
            realprofile = realprofile / lmaxfr
            # update norm after normalization
            norma = np.sum(realprofile**2)*dV

            if (deig < relax_eig_tol) & (verbose_level > 1):
                print(colored(" Linear convergence reached", 'green'))
                break

            if (iterl >= relax_iter_linear) & (verbose_level > 1):
                print(colored(" Linear convergence missed", 'red'))
                break

            iterl = iterl+1

        peak = np.max(realprofile)
        mean_variation = np.mean((realprofile-realprofile_old)**2)

        if (verbose_level > 1):
            print('Relaxation step, total, norm, peak, eig, conv, % d, % d,'
                  '% 5.2f, % 5.2f, % 5.2e, % 5.2e '
                  % (iternl, relax_niter,
                     norma, peak, eigenvalue, mean_variation))

        if (iternl >= relax_niter):
            print(colored("Missed nonlinear convergence", 'red'))
            break

        if (mean_variation < relax_tol):
            print(colored("Reached nonlinear convergence", 'green'))
            break

        iternl = iternl+1

    # store final guess
    psi0 = np.copy(realprofile)
    phi0 = np.copy(realprofile)

    return None


def plot_panels(psi, figure_name='./tests/img/test.png', titolo='titolo'):
    """ plot phi function sections in panels
    NLS.plot_panels(psi0, figure_name)
    Image.open(figure_name).show()

    Image.open(NLS.plot_panels(psi,figure_name)).show()

    Returns
    -------
    figure_name

    """
#    global x, y, t, nx, ny, nt
    matpsi0xy = np.squeeze(psi[:, :, nt//2])
    matpsi0xt = np.squeeze(psi[:, ny//2, :])
    matpsi0yt = np.squeeze(psi[nx//2, :, :])
#    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.pcolormesh(x, y, matpsi0xy, shading='auto')
    ax1.set(xlabel="x", ylabel="y")
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.pcolormesh(x, t, matpsi0xt, shading='auto')
    ax2.set(xlabel="x", ylabel="t")
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.pcolormesh(y, t, matpsi0yt, shading='auto')
    ax3.set(xlabel="y", ylabel="t")
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    [X, Y] = np.meshgrid(x, y)
    ax4.plot_surface(X, Y, matpsi0xy)
    fig.suptitle(titolo)
    fig.savefig(figure_name)
    plt.close()
    return figure_name


def saveall(filename='NLSdata'):
    """ save all data in a compressed file .npz """

    np.savez_compressed(filename,
                        nx=nx,
                        ny=ny,
                        nt=nt,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        tmin=tmin,
                        tmax=tmax,
                        nz=nz,
                        nplot=nplot,
                        zmax=zmax,
                        psi0=psi0,
                        phi0=phi0,
                        psi=psi,
                        phi=phi,
                        psi3D=psi3D,
                        phi3D=phi3D,
                        cxx=cxx,
                        cyy=cyy,
                        ctt=ctt,
                        chi=chi,
                        sigma_local=sigma_local,
                        sigma_nonlocal=sigma_nonlocal,
                        nphoton=nphoton,
                        alphanoise=alphanoise,
                        plot_level=plot_level,
                        verbose_level=verbose_level,
                        iter_implicit=iter_implicit,
                        relax_eig_tol=relax_eig_tol,
                        relax_tol=relax_tol,
                        relax_iter_linear=relax_iter_linear,
                        relax_alpha=relax_alpha,
                        relax_niter=relax_niter,
                        peaks=peaks,
                        powers=powers,
                        mean_pos=mean_pos,
                        mean_square_pos=mean_square_pos
                        )
    return None


def loadall(filename='NLSdata'):
    """ load all data in a compressed file .npz """

    global psi0, phi0
    global psi, phi
    global psi3D, phi3D
    global nx, ny, nt, nz, nplot, zmax
    global xmin, xmax
    global ymin, ymax
    global tmin, tmax
    global cxx, cyy, ctt
    global chi
    global sigma_local, sigma_nonlocal
    global alphanoise, nphoton
    global verbose_level, plot_level
    global iter_implicit
    global relax_tol, relax_alpha, relax_eig_tol
    global relax_iter_linear, relax_niter
    global peaks
    global mean_pos
    global mean_square_pos

    with np.load(filename) as data:
        phi = data['phi']
        psi = data['psi']
        phi0 = data['phi0']
        psi0 = data['psi0']
        phi3D = data['phi3D']
        psi3D = data['psi3D']
        cxx = data['cxx'].item()
        cyy = data['cyy'].item()
        ctt = data['cyy'].item()
        nx = data['nx'].item()
        ny = data['ny'].item()
        nt = data['nt'].item()
        xmin = data['xmin'].item()
        xmax = data['xmax'].item()
        ymin = data['ymin'].item()
        ymax = data['ymax'].item()
        tmin = data['tmin'].item()
        tmax = data['tmax'].item()
        nz = data['nz'].item()
        nplot = data['nplot'].item()
        zmax = data['zmax'].item()
        chi = data['chi'].item()
        sigma_local = data['sigma_local'].item()
        sigma_nonlocal = data['sigma_nonlocal'].item()
        iter_implicit = data['iter_implicit'].item()
        nphoton = data['nphoton'].item()
        alphanoise = data['alphanoise'].item()
        relax_alpha = data['relax_alpha'].item()
        relax_niter = data['relax_niter'].item()
        relax_iter_linear = data['relax_iter_linear'].item()
        relax_tol = data['relax_tol'].item()
        relax_eig_tol = data['relax_eig_tol'].item()
        peaks = data['peaks']
        mean_pos = data['mean_pos']
        mean_square_pos = data['mean_square_pos']

    # call initi to generate auxiliary vars
    init()

    return None


def plot_observables(figure_name='./tests/img/observables.png', titolo='observables'):
    """ plot observables """

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    for ishot in range(nshots):

        mean_xs = np.squeeze(mean_pos[0, :, ishot])
        mean_ys = np.squeeze(mean_pos[1, :, ishot])
        mean_ts = np.squeeze(mean_pos[2, :, ishot])
        mean_square_xs = np.squeeze(mean_square_pos[0, :, ishot])
        mean_square_ys = np.squeeze(mean_square_pos[1, :, ishot])
        mean_square_ts = np.squeeze(mean_square_pos[2, :, ishot])
        mean_square_rs = np.squeeze(mean_square_pos[3, :, ishot])

        ax1.plot(zplot, mean_xs)
        ax2.plot(zplot, mean_ys)
        ax3.plot(zplot, mean_ts)
        ax4.plot(zplot, mean_square_xs)
        ax5.plot(zplot, mean_square_ys)
        ax6.plot(zplot, mean_square_ts)
        ax6.plot(zplot, mean_square_rs)

    ax1.set(xlabel='z', ylabel='<x>')
    ax2.set(xlabel='z', ylabel='<y>')
    ax3.set(xlabel='z', ylabel='<t>')
    ax4.set(xlabel='z', ylabel='<x2>')
    ax5.set(xlabel='z', ylabel='<y2>')
    ax6.set(xlabel='z', ylabel='<t2, r2>')

    fig.suptitle(titolo)
    fig.savefig(figure_name)
    plt.close()

    return figure_name
