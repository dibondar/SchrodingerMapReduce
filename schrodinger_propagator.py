import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

###############################################################
class System:

    def U(self, x):
        """
        Potential
        :param x:
        :return:
        """
        return 0 # 0.5*(x)**2

    def F(self, x):
        """
        Force
        :param x:
        :return:
        """
        return -x

    def averages(self):
        x_rho = np.abs(self.wavefunc)**2
        x_rho /= x_rho.sum()
        av_x = np.dot(x_rho, self.x)
        av_x2 = np.dot(x_rho, self.x**2)

        p_rho = np.abs(fftpack.fft(self.wavefunc))**2
        p_rho /= p_rho.sum()
        av_p = np.dot(p_rho, self.p)
        av_p2 = np.dot(p_rho, self.p**2)

        return av_x, np.dot(self.force, x_rho), av_p, np.sqrt((av_x2-av_x**2)*(av_p2-av_p**2))

    def __init__(self, **kwargs):
        """
        Initialize system's parameters
        :param N: number of points to be used
        :param xmax: coordinate amplitude
        :return:
        """
        # Set parameters
        for k, v in kwargs.items():
            setattr(self, k, v)
        #
        # Coordinate range
        self.x = np.linspace(-self.xmax, self.xmax, self.N)
        self.dx = self.x[1]-self.x[0]
        #
        # Momentum range
        self.p = 2*np.pi*fftpack.fftfreq(self.x.size, self.dx)
        #
        # Verify that the force indeed corresponds to the -derivative of potential
        """
        if not np.allclose(-np.gradient(self.U(self.x), self.dx), self.F(self.x), rtol=1e-1):
            raise RuntimeError("Potential and Force are not compatible")
        """
        #
        # Define auxiliary arrays for split-operator
        self.expKE = np.exp(-0.5j*self.dt*self.p**2/self.m)
        #
        self.p_filter = np.exp(-2.*self.dt*0.5*self.p**2)
        #
        self.expU = np.exp(-1.j*self.dt*self.U(self.x))
        #
        self.force = self.F(self.x)

###############################################################

class SplitOperator(System):

    def propagate(self):
        """
        Propagate
        :param wavefunc:
        :return:
        """
        self.wavefunc *= self.expU
        self.wavefunc = fftpack.fft(self.wavefunc)
        self.wavefunc *= self.expKE
        self.wavefunc *= self.p_filter
        self.wavefunc = fftpack.ifft(self.wavefunc, overwrite_x=True)
        self.wavefunc /= np.linalg.norm(self.wavefunc)

###############################################################

class NewPropagator(System):
    """
    Define
        self.sigma
    """
    def propagate(self):
        #
        alpha = -0.5/self.sigma**2
        # propagation by expKE
        #
        c = 1. - 2j*self.dt*alpha
        #
        self.wavefunc /= np.sqrt(c)
        alpha = alpha / c
        #
        ##############################
        #
        # Lines below are equivalent to
        # ampl = sum(A*np.exp(alpha*(x-x0)**2) for A, x0 in izip(ampl, mu))
        #
        wavefunc = self.wavefunc[:,np.newaxis]
        mu =self.x[:,np.newaxis]
        x = self.x[np.newaxis,:]
        self.wavefunc = (wavefunc*np.exp(alpha*(x-mu)**2)).sum(axis=0)
        #
        # K = np.ceil(3*np.sqrt(-0.5/alpha.real) / self.dx)
        #
        ##############################
        #
        self.wavefunc *= self.expU
        self.wavefunc /= np.linalg.norm(self.wavefunc)

###############################################################
#
#   Initialize simulations
#
###############################################################

params = dict(dt=0.01, N=256, xmax=5, m=1, sigma=0.05) #sigma=0.03

SOp = SplitOperator(**params)
# Initial condition
SOp.wavefunc = np.exp(-(SOp.x+1)**2 + 1j*SOp.x) + 0j
SOp.wavefunc /= np.linalg.norm(SOp.wavefunc)

New = NewPropagator(**params)
# Initial condition
New.wavefunc = np.exp(-(New.x+1)**2 + 1j*New.x) + 0j
New.wavefunc /= np.linalg.norm(New.wavefunc)

###############################################################
#
#   Propagate
#
###############################################################

SOp_evolution = []
SOp_averages = []

New_evolution = []
New_averages = []

overlap = []

for i in xrange(1000):
    SOp.propagate()
    SOp_averages.append(SOp.averages())

    New.propagate()
    New_averages.append(New.averages())

    if i % 1 == 0:
        #print i
        SOp_evolution.append(SOp.wavefunc)
        New_evolution.append(New.wavefunc)

New_evolution.pop()

###############################################################
#
#   Plot
#
###############################################################

def plot_Ehrenfest(propagator, averages):
    dt = propagator.dt
    # Unpack averages
    av_x, av_force, av_p, uncertanty = [np.array(x) for x in zip(*averages)]
    #
    dx_dt = np.gradient(av_x, dt)
    # Effective mass
    m = 1./np.linalg.lstsq(av_p[:,np.newaxis], dx_dt)[0]


    print "Effective mass ", m
    plt.subplot(211)
    plt.plot(m*dx_dt)
    plt.plot(av_p)
    plt.title("First Ehrenfest theorem")

    dp_dt = np.gradient(av_p, dt)
    # Effective friction and spring constants
    gamma, f = np.linalg.lstsq(np.array([av_p, av_force]).T, dp_dt)[0]

    print "Effective friction constant ", gamma
    print "Force factor ", f

    plt.subplot(212)
    plt.plot(dp_dt, label='$d\\langle p \\rangle/dt$')
    plt.plot(gamma*av_p + f*av_force, label='$\\langle \\gamma p + f F(x) \\rangle$')
    #plt.plot(gamma*av_p + f*force(av_x), label='$\\gamma\\langle p\\rangle + f F(\\langle x\\rangle) $')
    plt.legend(loc='upper right')


#plot_Ehrenfest(SOp, SOp_averages)

plt.subplot(211)
plt.imshow(np.abs(np.array(SOp_evolution).T), origin='lower')
plt.title('Split-operator evolution')

plt.subplot(212)
plt.imshow(np.abs(np.array(New_evolution).T), origin='lower')
plt.title('New evolution')

plt.show()


