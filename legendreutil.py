import time

import numpy as np

from scipy.special import legendre
from numpy.polynomial.legendre import legvander
from numpy.polynomial import polyutils

import testutil

def get_polys(m) :
    return [legendre(i)*np.sqrt((2*i + 1)/2) for i in range(m)]

def test_polys(m) :
    """ test orthonormality """
    polys = get_polys(m)
    for i in range(len(polys)) :
        antid = (polys[i] * polys[i]).integ()
        testutil.assert_close(antid(1)-antid(-1), 1) # supposed to be 1
        for j in range(i+1, len(polys)) :
            antid = (polys[i] * polys[j]).integ()
            testutil.assert_close(antid(1)-antid(-1), 0) # supposed to be 0

def get_integrated_products(m, x) :

    p_x = legvander(np.array([x]), m+2)[0].T
    d_x = np.zeros(p_x.shape)
    d_x[1] = 1
    for n in range(2,len(p_x)) :
        d_x[n] = ((2*n-1) * x * d_x[n-1] - n * d_x[n-2])/(n-1)

    normalization_weights = [np.sqrt((2*i + 1)/2) for i in range(len(p_x))]

    p_x = np.multiply(p_x, normalization_weights)
    d_x = np.multiply(d_x, normalization_weights)

    res = np.zeros((m+1,m+1))
    range_list = np.array([i for i in range(m+1)])

    # fill upper and lower triangle
    for i in range(m+1) :
        res[i,i+1:] = (1-x**2) * (p_x[i]*d_x[i+1:m+1] - p_x[i+1:m+1]*d_x[i]) / (i+range_list[i+1:]+1) / (i-range_list[i+1:])
        res[i+1:,i] = res[i,i+1:]

    # fill diagonal
    res[0,0] = (x + 1)/2
    res[1,1] = (x**3 + 1)/2
    for i in range(2,m) :
        res[i,i] = res[i-1,i-1] + res[i+1, i-1] * (i+1) * np.sqrt(2*i - 1) / i / np.sqrt(2*i + 3) - res[i, i-2] * (i-1) * np.sqrt(2*i + 1) / i / np.sqrt(2*i - 3)

    return p_x[:-1], res[:-1, :-1]

def evaluate_basis(x, multiset) :
    """
    x : np.array with shape=(d,n)
    """

    indices = [multiset.IndexToMulti(i).GetVector() for i in range(multiset.Size())]
    m = len(indices)
    n = x.shape[1]
    d = x.shape[0]

    van = legvander(x, max([max(i) for i in indices]))

    mat = np.zeros((n, m))
    for i in range(n) :
        for j in range(m) :
            mat[i,j] = np.prod([van[k,i,indices[j][k if k < len(indices[j]) else 0]] for k in range(d)])
            mat[i,j] *= np.prod([np.sqrt((2*i + 1)/2) for i in indices[j]])
    return mat

def test_integrated_products(m, x) :
    res = get_integrated_products(m, x)[1]
    polys = get_polys(m)
    for i in range(m) :
        for j in range(i,m) :
            antid = np.polymul(polys[i],polys[j]).integ()
            val = antid(x) - antid(-1)
            testutil.assert_close(val, res[i,j])
            testutil.assert_close(val, res[j,i])

if __name__ == '__main__' :

    start = time.process_time()
    test_polys(20)
    for x in np.linspace(-1,1,123) :
        test_integrated_products(20, x)
    print('Time: ', time.process_time() - start)
