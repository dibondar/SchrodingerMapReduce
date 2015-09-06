import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from itertools import chain

class AbsSchrodingerPropagator:
    """
    Parameters to be defined

    self.dt -- time increment
    self.m -- mass
    self.x -- coordinate
    self.expU
    self.force
    """
    def averages(self, wavefunc, Nx=1, Np=1):
        """
        #
        averages = []
        #
        x_rho = np.abs(wavefunc)**2
        x_pow = np.ones_like(self.x)
        for n in xrange(Nx):
            x_pow *= self.x
            averages.append(np.sum(x_pow*x_rho))
        #
        p_wavefunc = fftpack.fft(wavefunc)
        p_wavefunc /= np.linalg.norm(p_wavefunc)
        p_rho = np.abs(p_wavefunc)**2
        #
        p_pow = np.ones_like(self.p)
        for n in xrange(Np):
            p_pow *= self.p
            averages.append(np.sum(p_pow*p_rho))
        #
        return tuple(averages)
        """
        # dx = self.x[1] - self.x[0]
        #
        x_rho = np.abs(wavefunc)**2
        x_rho /= x_rho.sum()
        av_x = np.sum(x_rho*self.x)
        av_x2 = np.sum(x_rho*self.x**2)

        p_rho = np.abs(fftpack.fft(wavefunc))**2
        p_rho /= p_rho.sum()
        av_p = np.sum(p_rho*self.p)
        av_p2 = np.sum(p_rho*self.p**2)

        return av_x, np.sum(self.force*x_rho), av_p, np.sqrt((av_x2-av_x**2)*(av_p2-av_p**2))

    def propagate(self, wavefunc):
        #
        ampl = wavefunc
        mu = self.x
        #
        #if alpha is None:
        # defying alpha based on mu
        sigma = 3.5*(mu[1]-mu[0])
        alpha = -0.5/sigma**2
        #
        #
        #ampl = np.linalg.lstsq(
        #    np.array([np.exp(alpha*(x-x0)**2) for x0 in x]).T,
        #    wavefunc
        #)[0]
        ##############################
        # propagation by expKE
        ##############################
        #
        c = 1. - 2j*self.dt*alpha
        #
        ampl = ampl / np.sqrt(c)
        alpha = alpha / c
        #
        ##############################
        #
        # Lines below are equivalent to
        # ampl = sum(A*np.exp(alpha*(x-x0)**2) for A, x0 in izip(ampl, mu))
        #
        ampl = ampl[np.newaxis,:]
        mu = mu[np.newaxis,:]
        x = self.x[:,np.newaxis]
        ampl = (ampl*np.exp(alpha*(x-mu)**2)).sum(axis=1)
        #
        ##############################
        #
        ampl *= self.expU
        ampl /= np.linalg.norm(ampl)
        #
        #self.alpha = alpha.real
        return ampl

# time increment
dt = 0.01

# coordinate range
x = np.linspace(-5, +5, 512)

alpha = 0.1
beta = 0.2

# potential
def U(x):
    return alpha*x**4 + beta*x**2

# force
def force(x):
    return -4*alpha*x**3 - 2*beta*x

expU = np.exp(-1j*dt*U(x))


class CHarmonicOscilator(AbsSchrodingerPropagator):
    dt = dt
    m = 1
    x = x
    p = 2*np.pi*fftpack.fftfreq(len(x), x[1]-x[0])
    expU = expU #(np.array([1.0 + 0j]), np.array([-1j*dt]), np.array([0j]))
    force = force(x)

HarmonicOsc = CHarmonicOscilator()

wavefunc = np.exp(-(x-3)**2) + 0j
wavefunc /= np.linalg.norm(wavefunc)

"""
plt.plot(x, np.abs(wavefunc)**2, label='initial')

for i in xrange(100):
    wavefunc = HarmonicOsc.propagate(wavefunc)

plt.plot(x, np.abs(wavefunc)**2, label='final')
plt.legend()
"""

averages = []
evolution = []

for i in xrange(2500):

    averages.append(HarmonicOsc.averages(wavefunc, Nx=3, Np=1))
    wavefunc = HarmonicOsc.propagate(wavefunc)

    if i % 3 == 0:
        #print i
        evolution.append(
            wavefunc
        )

plt.subplot(411)
plt.imshow(np.abs(np.array(evolution).T)**2)

plt.subplot(412)

#av_x, av_x2, av_x3, av_p = [np.array(x) for x in zip(*averages)]
av_x, av_force, av_p, uncertanty = [np.array(x) for x in zip(*averages)]


dx_dt = np.gradient(av_x, dt)
# Effective mass
m = 1./np.linalg.lstsq(av_p[:,np.newaxis], dx_dt)[0]

print "Effective mass ", m
plt.plot(m*dx_dt)
plt.plot(av_p)
plt.title("First Ehrenfest theorem")

plt.subplot(413)

dp_dt = np.gradient(av_p, dt)
# Effective friction and spring constants
gamma, f = np.linalg.lstsq(np.array([av_p, av_force]).T, dp_dt)[0]

print "Effective friction constant ", gamma
print "Force factor ", f

plt.plot(dp_dt, label='$d\\langle p \\rangle/dt$')
plt.plot(gamma*av_p + f*av_force, label='$\\langle \\gamma p + f F(x) \\rangle$')
#plt.plot(gamma*av_p + f*force(av_x), label='$\\gamma\\langle p\\rangle + f F(\\langle x\\rangle) $')
plt.legend()

plt.subplot(414)
plt.plot(uncertanty)

plt.show()




