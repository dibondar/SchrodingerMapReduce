import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from scipy import fftpack

import numpy as np

class System:
    #
    #
    # Rules of converting python types to CUDA types
    python2CUDA = {
        float : "__constant__ float %s = %f;\n",
        int : "__constant__ int %s  = %d;\n"
    }
    #
    # CUDA code
    #
    CUDA_propagator = """
        #include<pycuda-complex.hpp>
        #include<math.h>
        #define _USE_MATH_DEFINES

        typedef pycuda::complex<float> cuda_complex;

        // Constants
        {CUDA_constants}

        __global__ void unormalize_propagate(cuda_complex *wavefunc_in, cuda_complex *wavefunc_out)
        {{
            const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x;

            // Grid step size
            const float dx = 2.*xmax / float(N);

            // width of the Gaussian before propagation
            const float alpha_in = -0.5/(sigma*sigma);

            // scaling factor
            const cuda_complex c = cuda_complex(1., - 2.*dt*alpha_in);

            // width of the Gaussian after propagation
            const cuda_complex alpha_out = cuda_complex(alpha_in, 0.) / c;

            // const float x = -xmax +  indexTotal *2.*xmax/float(N-1.);

            cuda_complex result = wavefunc_in[indexTotal];

            for (int k = 1; k <=  min(indexTotal, K); k++)
                result += wavefunc_in[indexTotal-k]* exp(alpha_out*dx*dx*float(k*k));

            for (int k = 1; k <=  min(N-indexTotal-1, K); k++)
                result += wavefunc_in[indexTotal+k] * exp(alpha_out*dx*dx*float(k*k));

            wavefunc_out[indexTotal] = result / sqrt(c);
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

    def cuda_propagate(self, wavefunc_in, wavefunc_out):
        #
        # Propagate without normalization
        self.unormalize_propagate(wavefunc_in, wavefunc_out, block=(1,1,1), grid=(self.N,1))
        #
        # normalize output
        wavefunc_out /= np.sqrt(gpuarray.dot(wavefunc_out, wavefunc_out).get())


###############################################################

class NewPropagator(System):
    pass

###############################################################
#
#   Initialize simulations
#
###############################################################

params = dict(dt=0.01, N=1024, xmax=10, m=1, sigma=0.05, K=20) #sigma=0.03

New = NewPropagator(**params)
x = New.x

wavefunc_in = np.exp(-1*x**2) + 0j
wavefunc_in /= np.linalg.norm(wavefunc_in)
wavefunc_in = gpuarray.to_gpu(np.ascontiguousarray(wavefunc_in , dtype=np.complex64))
wavefunc_out = gpuarray.zeros_like(wavefunc_in)

plt.plot(x, np.abs(wavefunc_in.get()), label='initial')

for t in xrange(50):
    New.cuda_propagate(wavefunc_in, wavefunc_out)
    wavefunc_in, wavefunc_out = wavefunc_out, wavefunc_in

plt.plot(x, np.abs(wavefunc_in.get()), label='out')
plt.legend()
plt.show()



