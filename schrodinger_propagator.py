__author__ = 'kdfstudio'

import numpy as np
from collections import defaultdict
from itertools import chain, izip

def MapReduce(mapper, reducer, input_data):
    """
    Simple implementation of map reducer
    :param mapper: a mapper generator must yield (key, value)
    :param reducer: a reducer generator
    :return: processed data

    # Character count example using MaoReduce
    def mapper(x):
        yield (x, 1)

    def reducer(k, v):
        yield (k, sum(v))

    print MapReduce(mapper, reducer, "TestInputString")
    """
    # Map data
    if mapper is None:
        mapped_data = input_data
    else:
        mapped_data = chain(*(mapper(x) for x in input_data))
    #
    if reducer is None:
        return list(mapped_data)
    #
    # Shuffle data
    shuffled_data = defaultdict(list)
    for key, value in mapped_data:
        shuffled_data[key].append(value)
    #
    # Reduce data
    reduced_data = (reducer(k, v) for k, v in shuffled_data.items())
    reduced_data = list(chain(*reduced_data))
    #
    return reduced_data

#################################################################

class AbsSchrodingerPropagator:
    """
    Parameters to be defined

    self.dt -- time increment
    self.m -- mass
    self.expUexpansion -- interpolation of exp(-1j*dt*U(x)) with ampl*exp(alpha*x^2 + beta*x)
    self.weight_cutoff --
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
        :param gaussian: (ampl, alpha, beta) tuple denoting ampl*exp(alpha*x^2 + beta*x)
        :return: gaussian-like tuple
        """
        # unpack the gaussian in coordinate representation
        ampl, alpha, beta = gaussian
        #
        c = 1 - 2j*self.dt*alpha
        #
        ampl *= np.exp(1j*self.dt*beta**2/(4*self.m*c)) / np.sqrt(c)
        alpha /= c
        beta /= c
        #
        yield (ampl, alpha, beta)

    def expU_Mapper(self, gaussian):
        """
        Calculate action of exp(-1j*dt*U(x)) on a gaussian
        given in the coordinate representation
        :param gaussian: (ampl, alpha, beta) tuple denoting ampl*exp(alpha*x^2 + beta*x)
        :return: gaussian-like tuples
        """
        ampl, alpha, beta = gaussian
        #
        for ampl_, alpha_, beta_ in self.expUexpansion:
            yield (ampl + ampl_, alpha * alpha_, beta * beta_)

    def getnorm_Mapper(self, gaussian):
        """
        Calculate normalization integral
            N = int(abs(ampl*exp(alpha*x^2 + beta*x))**2 , x=-infinity..+infinity)
        :param gaussian: (ampl, alpha, beta) tuple denoting ampl*exp(alpha*x^2 + beta*x)
        :return: N
        """
        ampl, alpha, beta = gaussian
        #
        if np.real(alpha) > 0:
            print "Warning: Divergent integral of ", gaussian
            N = np.inf
        else:
            ra = np.real(alpha)
            rb = np.real(beta)
            N = np.abs(ampl)**2 * np.sqrt(-0.5*np.pi/ra) * np.exp(-0.5*rb**2/ra)
        #
        yield N

    #############################################################
    #
    #   Propagator
    #
    #############################################################

    def propagate(self, wavefunc):
        """
        Propagate wave function by one time step using
        the Trotter product
        :return:
        """
        #
        # propagate by kinetic part
        wavefunc = MapReduce(self.expKE_Mapper, None, wavefunc)
        #
        # propagate by potential part
        wavefunc = MapReduce(self.expU_Mapper, None, wavefunc)
        #
        # get norms of each gaussian
        norms = MapReduce(self.getnorm_Mapper, None, wavefunc)
        #
        norms = np.array(norms)
        norms /= norms.sum()
        #
        wavefunc = [g for g, n in izip(wavefunc, norms) if n > self.weight_cutoff]
        #
        return wavefunc

#################################################################
#
# Harmonic oscilator propagation
#
#################################################################

# time increment
dt = 0.01

# coordinate range
x = np.linspace(-5, +5, 100)

# potential
U = 0.5 * x**2
expU = np.exp(-1j*dt*U)

# approximate expU by gaussians
sigma = 0.5 * (x[1]-x[0])
alpha = -0.5/sigma**2

coeffs = np.linalg.lstsq(
    [np.exp(alpha*(x-x0)**2) for x0 in x],
    expU
)[0]

# Check the difference
print "Accuracy of solving lin equations ", \
np.linalg.norm(
    expU -
    sum(c*np.exp(alpha*(x-x0)**2) for c, x0 in izip(coeffs, x))
)


expUexpansion = [
    (c*np.exp(alpha*x0**2), alpha, -2*x0*alpha)
    for c, x0 in izip(coeffs, x)
]

# remove terms with zero coefficients
expUexpansion = [g for g in expUexpansion if g[0]]

# Check the difference
print "Accuracy of the approximation ", \
np.linalg.norm(
    expU -
    sum(ampl*np.exp(alpha*x**2 + beta*x) for ampl, alpha, beta in expUexpansion)
)



