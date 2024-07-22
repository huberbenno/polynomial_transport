import numpy as np
import scipy.special as ss
import itertools

import MultiIndex as mi
from . import random, legendre


def scale(x, d1, d2) :
    """
    Affine transformation from the interval d1 to the interval d2 applied to point x.
    x : scalar or array or list of shape (n, d) or (d,)
    d1, d2 : arrays or lists of shape (d, 2) or (2,)
    """
    #if d1 is None : d1 = [-1,1]
    #if d2 is None : d2 = self.domain

    # ensure d1, d2 have shape (d, 2)
    d1, d2 = np.array(d1), np.array(d2)
    assert d1.shape == d2.shape
    d = 1 if len(d1.shape) == 1 else d1.shape[0]
    if len(d1.shape) == 1 :
        d1 = d1.reshape((1,2))
        d2 = d2.reshape((1,2))
    for i in range(d) :
        assert d1[i,0] < d1[i,1]
        assert d2[i,0] < d2[i,1]

    # ensure x has shape (n, d)
    x = np.array(x)
    x_shape = x.shape
    if x_shape == () :
        x = np.array([[x]])
    else :
        x = np.array(x)
        if len(x.shape) == 1 :
            if x.shape[0] == d :
                x = x.reshape((1,len(x)))
            else :
                x = x.reshape((len(x),1))

    # ensure x in d1
    for i in range(d) :
        for j in range(x.shape[0]) :
            assert (x[j,i] >= d1[i,0] or np.isclose(x[j,i], d1[i,0])),\
                    f'Assertion failed with\n x[{j},{i}] ({x[j,i]})\n d1[{i},0] ({d1[i,0]})'
            assert (x[j,i] <= d1[i,1] or np.isclose(x[j,i], d1[i,1])),\
                    f'Assertion failed with\n x[{j},{i}] ({x[j,i]})\n d1[i,1] ({d1[i,1]})'

    # check
    assert len(x.shape) == len(d1.shape) == len(d2.shape) == 2
    assert x.shape[1] == d1.shape[0] == d2.shape[0]
    assert d1.shape[1] == d2.shape[1] == 2

    # scale
    for i in range(d) :
        x[:,i] = (x[:,i] - d1[i,0]) / (d1[i,1] - d1[i,0])
        x[:,i] = x[:,i] * (d2[i,1] - d2[i,0]) + d2[i,0]

    # ensure x in d2
    #for i in range(d) :
    #    assert (x[:,i] >= d2[i,0]).all(), f"Assertion failed with\n x[:,i] ({x[:,i]})\n d2[i,0] ({d2[i,0]})"
    #    assert (x[:,i] <= d2[i,1]).all(), f"Assertion failed with\n x[:,i] ({x[:,i]})\n d2[i,1] ({d2[i,1]})"

    # return in original shape
    return x.reshape(x_shape)


def ensure_shape(x, d) :
    """ ensures that x is of shape (d, n) """
    if isinstance(x, float) :
        assert d == 1
        return np.array([[x]])
    if isinstance(x, list) :
        x = np.array(x)
    assert x.ndim <= 2
    if x.ndim == 0 :
        assert d == 1
        return np.array([[x]])

    if x.ndim == 1 :
        if d == 1 :
            return np.expand_dims(x, axis=0)
        else :
            assert x.shape[0] == d
            return np.expand_dims(x, axis=1)
    if x.ndim == 2 and x.shape[0] != d :
        return x.T
    return x


def bisection(f, y, *, interval=(-1,1), n=30) :
    midpoint = lambda interval : interval[0] + (interval[1] - interval[0])/2
    x = midpoint(interval)
    candidate = f(x)
    for _ in range(n) :
        if   candidate > y : interval = [interval[0],  x]
        elif candidate < y : interval = [x, interval[1]]
        else : break
        x = midpoint(interval)
        candidate = f(x)
    return x


def chebychev_1d(n) :  # https://en.wikipedia.org/wiki/Chebyshev_nodes
    return np.array([np.cos((2*k+1)*np.pi/2/n) for k in range(n)])


def leja_1d(n) :
    r = [1]
    if n > 0 : r.append(-1)
    if n > 1 : r.append(0)
    for j in range(3, n+1) :
        if j % 2 == 0 :
            r.append(-r[j-1])
        else :
            r.append(np.sqrt((r[int((j+1)/2)] + 1) / 2))
    return np.array(r)


def leja(multiset) :
    r = leja_1d(multiset.maxDegree)
    p = np.zeros((multiset.dim, multiset.cardinality))
    for i in range(multiset.dim) :
        for j in range(multiset.cardinality) :
            p[i,j] = r[multiset[j][i]]
    return p


def leggaus(multiset) :
    p1d, w1d = np.polynomial.legendre.leggauss(multiset.maxDegree + 1)
    p = np.zeros((multiset.dim, multiset.cardinality))
    w = np.zeros((multiset.dim, multiset.cardinality))
    for i in range(multiset.dim) :
        for j in range(multiset.cardinality) :
            p[i,j] = p1d[multiset[j][i]]
            w[i,j] = w1d[multiset[j][i]]

    #w = np.multiply(np.prod(w, axis=0), np.array([(-1.)**(multiset.maxDegree - np.sum(idx)) * ss.binom(multiset.maxDegree, np.sum(idx)) for idx in multiset.asLists()]))
    #print(np.sum(w))
    return p, None #np.sum(w, axis=0)/np.sum(w)


def cheby_weights(points) :
    return (np.pi/2)**points.shape[0] * np.prod(np.sqrt(1-points**2), axis=0)


def get_sample_points_and_weights(multis, dist, n) :
    if dist == 'uni' :
        return np.random.uniform(low=-1, high=1, size=(multis.dim, n)), None
    if dist == 'leja' :
        return leja(multis), None
    elif dist == 'leggaus' :
        return leggaus(multis)
    if dist == 'cheby' :
        points = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2, size=(multis.dim, n)))
        return points, cheby_weights(points)
    if dist == 'cheby_ss' :
        import julia
        julia.Main.include("../BSSsubsampling.jl/src/BSSsubsampling.jl")
        points = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2, size=(multis.dim, 2*n)))
        Y = legendre.evaluate_basis(points, multis)
        idxs, s = julia.Main.BSSsubsampling.bss(Y, n/multis.size(), A=1, B=1)
        return points[:, idxs-1], cheby_weights(points)
    if dist == 'christoffel' :
        polys = [ss.legendre(i)*np.sqrt((2*i + 1)) for i in range(multis.maxDegree+1)]
        polys = [(p*p).integ() for p in polys]
        d = multis.dim
        m = multis.cardinality
        points = np.random.uniform(low=-1, high=1, size=(d, n))
        weights = np.zeros((n,))
        for j in range(n) :
            idx = multis[np.random.randint(low=0, high=m)].asList()
            for i in range(d) :
                points[i,j] = bisection(polys[idx[i]], points[i,j])
            weights[j] = m/np.sum([np.prod([polys[k](points[l,j])**2 for l,k in enumerate(multis[q].asList())]) for q in range(multis.size())])
        return points, weights


def get_sample_points_and_weights_deterministic(multis, dist, n='wls') :
    """ !!! EXPERIMENTAL !!! """
    points = {'cheby_det' : chebychev_1d, 'leja' : leja_1d, 'leggaus' : leggaus}[dist]

    # TODO think through weights for leja or leggaus below!
    if dist == 'shuffled':
        res = []
        for i in range(multis.dim) :
            base = points(n)
            random.rng.shuffle(base)
            res.append(base)
        samples = np.array(res)
    elif dist == 'tp_light':
        res = list(itertools.product(points(int(np.ceil(n**(1/multis.dim)))), repeat=multis.dim))
        samples = np.array(random.rng.choice(res, size=n, replace=False)).T
    elif dist == 'sparse_grid' :
        samples = n.getSparseGrid(lambda kmax : np.array(points(kmax)))
    else : assert False

    return samples, cheby_weights(samples)
