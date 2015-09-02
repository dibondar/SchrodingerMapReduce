__author__ = 'kdfstudio'

import numpy as np
from itertools import chain, izip
import matplotlib.pyplot as plt

#################################################################

class AbsSchrodingerPropagator:
    """
    Parameters to be defined

    self.dt -- time increment
    self.m -- mass
    self.expUexpansion -- interpolation of exp(-1j*dt*U(x)) with ampl*exp(alpha*x^2 + beta*x)
    self.norm_cutoff --
    """
    #############################################################
    #
    #   MapReduce declarations
    #
    #############################################################

    def expKE_Mapper(self, gaussian):
        """
        Calculate action of exp(-1j*dt*p^2/(2m)) on
        a gaussian given in the coordinate representation
        :param gaussian: (ampl, alpha, mu) tuple denoting ampl*exp(alpha*(x-mu)**2)
        :return: gaussian-like tuple
        """
        # unpack the gaussian in coordinate representation
        ampl, alpha, mu = gaussian
        #
        c = 1 - 2j*self.dt*alpha
        #
        ampl = ampl / np.sqrt(c)
        alpha = alpha / c
        #
        return ampl, alpha, mu

    def expU_Mapper(self, gaussian):
        """
        Calculate action of exp(-1j*dt*U(x)) on a gaussian
        given in the coordinate representation
        :param gaussian: (ampl, alpha, mu) tuple denoting ampl*exp(alpha*(x-mu)**2)
        :return: gaussian-like tuples
        """
        return self.multiply(gaussian, self.expUexpansion)

    def multiply(self, gaussian1, gaussian2):
        #
        ampl_1, alpha_1, mu_1 = gaussian1
        #
        ampl_1  = ampl_1[:, np.newaxis]
        alpha_1 = alpha_1[:, np.newaxis]
        mu_1    = mu_1[:, np.newaxis]
        #
        ampl_2, alpha_2, mu_2 = gaussian2
        #
        ampl_2  = ampl_2[np.newaxis,:]
        alpha_2 = alpha_2[np.newaxis,:]
        mu_2    = mu_2[np.newaxis,:]
        #
        alpha = alpha_1 + alpha_2
        mu = (alpha_1*mu_1 + alpha_2*mu_2) / alpha
        ampl = ampl_1*ampl_2 * np.exp(alpha_1 * mu_1**2 + alpha_2 * mu_2**2 - alpha * mu**2)
        #
        return np.ravel(ampl), np.ravel(alpha), np.ravel(mu)

    def norms_Mapper(self, gaussian):
        """
        Calculate normalization integral
            N = int(abs(ampl*exp(alpha*(x-mu)**2))**2 , x=-infinity..+infinity)
        :param gaussian: (ampl, alpha, mu) tuple denoting ampl*exp(alpha*(x-mu)**2)
        :return: N
        """
        ampl, alpha, mu = gaussian
        #
        a = -2.*alpha.real
        b = -4*np.real(mu*alpha)
        #
        norms = np.abs(ampl)**2 * np.sqrt(np.pi/a) * np.exp(
                0.25*b**2/a + 2*np.real(alpha*mu**2)
        )
        #
        return norms

    #############################################################
    #
    #   Propagator
    #
    #############################################################

    @classmethod
    def getwavefunc(cls, gaussian, x):
        #
        ampl, alpha, mu = gaussian
        #
        ampl = ampl[np.newaxis,:]
        alpha = alpha[np.newaxis,:]
        mu = mu[np.newaxis,:]
        #
        x = x[:,np.newaxis]
        #
        return np.sum(
            ampl*np.exp(alpha*(x - mu)**2),
            axis=1
        )

    def propagate(self, wavefunc):
        """
        Propagate wave function by one time step using
        the Trotter product
        :return:
        """
        #
        # propagate by kinetic part
        wavefunc = self.expKE_Mapper(wavefunc)
        #
        # propagate by potential part
        wavefunc = self.expU_Mapper(wavefunc)
        #
        # propagate by kinetic part
        wavefunc = self.expKE_Mapper(wavefunc)
        #
        #norms = self.norms_Mapper(wavefunc)
        #norms /= norms.sum()
        #
        # find out which gausian to delete
        # indx = np.nonzero(norms < self.norm_cutoff)[0]
        #
        # wavefunc = [np.delete(param, indx) for param in wavefunc]
        #
        # print wavefunc[0].size
        #
        #
        # resumpling
        x = np.linspace(-5, +5, 100)

        sigma = 0.5 * (x[1]-x[0])
        alpha = -0.5/sigma**2

        psi = self.getwavefunc(wavefunc, x)
        psi /= np.linalg.norm(psi)

        ampls = np.linalg.lstsq(
            [np.exp(alpha*(x-x0)**2) for x0 in x],
            psi
        )[0]

        wavefunc = (ampls, alpha*np.ones_like(x), x)
        return wavefunc

#################################################################
#
# Harmonic oscilator propagation
#
#################################################################

# time increment
dt = 0.025

# coordinate range
x = np.linspace(-5, +5, 100)

# potential
U = 0.5 * x**4
expU = np.exp(-1j*dt*U)

# approximate expU by gaussians
sigma = 0.5 * (x[1]-x[0])
alpha = -0.5/sigma**2

coeffs = np.linalg.lstsq(
    [np.exp(alpha*(x-x0)**2) for x0 in x],
    expU
)[0]

# Check the difference
print "Accuracy of the approximation expU ", \
np.linalg.norm(
    expU -
    sum(c*np.exp(alpha*(x-x0)**2) for c, x0 in izip(coeffs, x))
)

expUexpansion = (coeffs, alpha*np.ones_like(x), x)

def ProbDensity(wave, x):
    psi = sum(
        c * np.exp(a*(x-mu)**2) for c, a, mu in izip(*wave)
    )
    return np.abs(psi)


class CHarmonicOscilator(AbsSchrodingerPropagator):
    dt = 0.5*dt
    m = 1
    expUexpansion = expUexpansion
    norm_cutoff = 1e-7

HarmonicOsc = CHarmonicOscilator()

wavefunc = ( np.array([1. + 0j]), np.array([-1. + 0j]), np.array([0 + 0.5j]) )

#plt.plot(x, np.abs(HarmonicOsc.getwavefunc(wavefunc, x))**2, label='initial condition')

density = []
for i in range(200):
    print i
    wavefunc = HarmonicOsc.propagate(wavefunc)
    density.append(
        np.abs(HarmonicOsc.getwavefunc(wavefunc, x))**2
    )

#plt.plot(x, np.abs(HarmonicOsc.getwavefunc(wavefunc1, x))**2, label='final condition')
#plt.legend()
plt.imshow(density)
plt.show()
