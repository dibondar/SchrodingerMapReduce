import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import skcuda.linalg
try:
    skcuda.linalg.init()
except ImportError:
    pass

import matplotlib.pyplot as plt
from scipy import fftpack

import numpy as np

class System:
    #
    #
    # Rules of converting python types to CUDA types
    python2CUDA = {
        float : "const float %s = %f;\n",
        int : "const int %s  = %d;\n"
    }
    #
    # CUDA code
    #
    CUDA_propagator = """
        #include<pycuda-complex.hpp>
        #include<math.h>
        #define _USE_MATH_DEFINES

        typedef pycuda::complex<float> cuda_complex;

        __global__ void unormalize_propagate(cuda_complex *wavefunc_in, cuda_complex *wavefunc_out)
        {{
            const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x;

            ////////////////////////////// Constants //////////////////////////////
            {CUDA_constants}

            // Grid step size
            const float dx = 2.*xmax / float(N);

            // width of the Gaussian before propagation
            const float alpha_in = -0.5/(sigma*sigma);

            // scaling factor
            const cuda_complex c = cuda_complex(1., - 2.*dt*alpha_in);

            // width of the Gaussian after propagation
            const cuda_complex alpha_out = cuda_complex(alpha_in, 0.) / c;

            // How many gaussians to add
            const int K = ceil( S*sqrt(abs(0.5/real(alpha_out))) / dx );

            // Curret coordinate
            const float x = -xmax +  indexTotal *2.*xmax/float(N-1.);
            ///////////////////////////////////////////////////////////////////////

            cuda_complex result = wavefunc_in[indexTotal];

            // propagation by np.exp(-0.5j*self.dt*self.p**2/self.m)
            for (int k = 1; k <=  min(indexTotal, K); k++)
                result += wavefunc_in[indexTotal-k]* exp(alpha_out*dx*dx*float(k*k));

            for (int k = 1; k <=  min(N-indexTotal-1, K); k++)
                result += wavefunc_in[indexTotal+k] * exp(alpha_out*dx*dx*float(k*k));

            // propagation by np.exp(-1.j*self.dt*self.potentialString)
            result *= exp(cuda_complex(0, -dt*({potentialString})));

            // Return the result
            wavefunc_out[indexTotal] = result;
        }}
    """
    #
    #
    def __init__(self, **kwargs):
        #
        # Constants to be added to CUDA code
        self.CUDA_constants = ""
        #
        # Set parameters
        for k, v in kwargs.items():
            setattr(self, k, v)
            try:
                self.CUDA_constants += self.python2CUDA[type(v)] %(k, v)
            except KeyError:
                pass
        #
        # Check whether user specified potential
        try:
            self.potentialString
        except AttributeError:
            print("\nWarning: potentialString has not been specified. Free particle is assumed.\n")
            self.potentialString = "0.0"
        #
        # Coordinate range
        self.x = np.linspace(-self.xmax, self.xmax, self.N)
        self.dx = self.x[1]-self.x[0]
        #
        # Momentum range
        self.p = 2*np.pi*fftpack.fftfreq(self.x.size, self.dx)
        #
        # Define auxiliary arrays for split-operator
        self.expKE = np.exp(-0.5j*self.dt*self.p**2/self.m)
        #
        # Define coordinate for eval
        x = self.x
        #
        self.expU = np.exp(-1.j*self.dt*eval(self.potentialString))
        #
        # CUDA code
        self.CUDA_propagator = self.CUDA_propagator.format(
            CUDA_constants=self.CUDA_constants, potentialString=self.potentialString
        )
        #
        # Compile code
        self.unormalize_propagate = SourceModule(self.CUDA_propagator).get_function("unormalize_propagate")
        #
        #
        alpha = -0.5/self.sigma**2
        # propagation by expKE
        #
        c = 1. - 2j*self.dt*alpha
        #
        alpha = alpha / c
        print self.S*np.ceil(np.sqrt(-0.5/alpha.real) / self.dx)


    def cuda_propagate(self, wavefunc_in, wavefunc_out):
        #
        # Propagate without normalization
        self.unormalize_propagate(wavefunc_in, wavefunc_out, block=(256,1,1), grid=(self.N/256,1))
        #
        # normalize output
        # wavefunc_out /= np.sqrt(gpuarray.dot(wavefunc_out.conj(), wavefunc_out).get())
        wavefunc_out /= skcuda.linalg.norm(wavefunc_out)

###############################################################
#
#   Initialize simulations
#
###############################################################

params = dict(dt=0.01, N=1024, xmax=10, m=1, sigma=0.03, S=10, potentialString='x*x') #sigma=0.03

New = System(**params)
x = New.x

# Initial condition
wavefunc_in = np.exp(-1*(x-3)**2) + 0j
wavefunc_in /= np.linalg.norm(wavefunc_in)
wavefunc_in = gpuarray.to_gpu(np.ascontiguousarray(wavefunc_in , dtype=np.complex64))
wavefunc_out = gpuarray.zeros_like(wavefunc_in)

###############################################################
#
#   Propagate
#
###############################################################

SOp_evolution = []

New_evolution = []

for i in xrange(10000):
    New.cuda_propagate(wavefunc_in, wavefunc_out)
    wavefunc_in, wavefunc_out = wavefunc_out, wavefunc_in
    if i % 10 == 0:
        New_evolution.append(abs(wavefunc_in).get())

###############################################################
#
#   Plot
#
###############################################################

plt.imshow(np.array(New_evolution).T, origin='lower')
plt.show()




