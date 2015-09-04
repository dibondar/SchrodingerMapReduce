__author__ = 'kdfstudio'
"""
Given linear combination of gaussians reduce redundancy
"""
import numpy as np
import matplotlib.pyplot as plt


def scalar_product(gaussian1, gaussian2):

    ampl1, alpha1, mu1 = np.conjugate(gaussian1)
    ampl2, alpha2, mu2 = gaussian2

    return ampl1*ampl2*  np.sqrt(-np.pi/(alpha1+alpha2))\
          * np.exp((alpha1*alpha2*(mu1-mu2)**2) / (alpha1+alpha2))


def GetSimilarity(gaussian1, gaussian2):
    return scalar_product(gaussian1, gaussian2) \
            / np.sqrt(scalar_product(gaussian1, gaussian1)*scalar_product(gaussian2, gaussian2))

"""
def GetSimilarity(gaussian1, gaussian2):
    diff_norm = scalar_product(gaussian2, gaussian2) + scalar_product(gaussian1, gaussian1) \
        - 2*np.real(scalar_product(gaussian1, gaussian2))
    diff_norm = np.sqrt(np.real(diff_norm))
    sum_norm = scalar_product(gaussian2, gaussian2) + scalar_product(gaussian1, gaussian1) \
        + 2*np.real(scalar_product(gaussian1, gaussian2))
    sum_norm = np.sqrt(np.real(sum_norm))
    return diff_norm/sum_norm
"""

def Merge(gaussian1, gaussian2):
    ampl1, alpha1, mu1 = gaussian1
    ampl2, alpha2, mu2 = gaussian2

    mu = (ampl1*mu1 + ampl2*mu2)/(ampl1 + ampl2)

    ampl = ampl1*np.exp(alpha1*(mu-mu1)) + ampl2*np.exp(alpha2*(mu-mu2))

    # http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
    SigmaSq = ((-0.5/alpha1 + mu1**2)*ampl1 + (-0.5/alpha2 + mu2**2)*ampl2)/(ampl1 + ampl2) - mu**2
    alpha = -0.5/SigmaSq
    if np.real(alpha) > 0:
        print "Warning: Divergent merge"
        norm1 = np.real(scalar_product(gaussian1, gaussian1))
        norm2 = np.real(scalar_product(gaussian2, gaussian2))
        if norm1 > norm2:
            return gaussian1
        else:
            return gaussian2
    else:
        return ampl, alpha, mu

def analyze_simplify(wavefunc):
    #
    equival_classes = []
    for g in wavefunc:
        new_class = True
        for c in equival_classes:
            if GetSimilarity(g, c[0]) > 0.99:
                c.append(g)
                new_class = False
                break
        if new_class:
            equival_classes.append([g])

    new_wavefunc = [reduce(Merge, c) for c in equival_classes]
    return new_wavefunc


# Number of gaussions to generate
Ng = 500

x = np.linspace(-15, 15, 1000)

state = np.random.get_state()

original_gaussians = zip(
    np.random.uniform(-3., 3, Ng) + 1j*np.random.uniform(-3., 3, Ng),
    np.random.uniform(-0.3, -0.2, Ng) + 1j*np.random.uniform(-1, 1, Ng),
    np.random.uniform(0.6*x.min(), 0.6*x.max(), Ng)
)

def SumGauss(gaussians):
    return sum(ampl*np.exp(alpha*(x-mu)**2) for ampl, alpha, mu in gaussians)

plt.plot(x, np.abs(SumGauss(original_gaussians)), label='sum')
print "Original length ", len(original_gaussians)

simplify_gaussians = original_gaussians
for i in range(1):
    simplify_gaussians = analyze_simplify(simplify_gaussians)
    print "Simplify length ", len(simplify_gaussians)

print np.linalg.norm(SumGauss(original_gaussians) - SumGauss(simplify_gaussians))

plt.plot(x, np.abs(SumGauss(simplify_gaussians)), label='merger')
plt.legend()
plt.show()
