import numpy as np
import scipy.special as ss
import itertools

#from julia import Main
#Main.include("../BSSsubsampling.jl/src/BSSsubsampling.jl")

import randutil, legendreutil

def bisection(f,y) :
    midpoint = lambda interval : interval[0] + (interval[1] - interval[0])/2
    interval = [-1,1]
    x = 0
    candidate = f(x)
    for _ in range(30) :
        if candidate > y : interval = [interval[0],  x]
        else :             interval = [x, interval[1]]
        x = midpoint(interval)
        candidate = f(x)
    return x

def chebychev_1d(n) : # https://en.wikipedia.org/wiki/Chebyshev_nodes
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
    return np.array(r).reshape((len(r),1))

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

def cheby_weights(samples) :
    return (np.pi/2)**samples.shape[0] * np.prod(np.sqrt(1-samples**2), axis=0)

def get_sample_points_and_weights_legacy(method, multis) :
    d = multis.dim
    m = multis.size()
    assert(m > 0)
    samples = None
    weights = None
    if method == 'ip_leja' :
        samples = leja(multis)
    elif method == 'ip_leggaus' :
        samples, weights = leggaus(multis)
    elif method == 'wls_leja' :
        k = 1
        if multis.dim == 1 :
            k = int(1.5 * multis.maxDegree)
        else :
            while ss.binom(k+1, multis.dim-1) <= 1.5 * multis.cardinality :
                k += 1
        m = mi.TotalDegreeSet(dim=multis.dim, order=k, save=False)
        samples = leja(m)
    elif method == 'wls_leggaus' :
        k = 1
        if multis.dim == 1 :
            k = int(1.5 * multis.maxDegree)
        else :
            while ss.binom(k+1, multis.dim-1) <= 1.5 * multis.cardinality :
                k += 1
        m = mi.TotalDegreeSet(dim=multis.dim, order=k, save=False)
        samples, weights = leggaus(m)
    else : assert(False)
    return samples, weights

def get_sample_points_and_weights_random(multis, mode, n) :
    if mode == 'uni' :
        return np.random.uniform(low=-1, high=1, size=(multis.dim, n)), None
    if mode == 'cheby' :
        samples = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2, size=(multis.dim, n)))
        return samples, cheby_weights(samples)
    if mode == 'cheby_ss' :
        samples = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2, size=(multis.dim, 2*n)))
        Y = legendreutil.evaluate_basis(samples, multis)
        idxs, s = Main.BSSsubsampling.bss(Y, n/multis.size(), A = 1, B = 1)
        #print(n, Y.shape, idxs.shape, multis.size())
        samples = samples[:, idxs-1]
        ws = cheby_weights(samples)
        #for w1,w2 in zip(s,ws) : print(w1, '\t', w2, '\t', w1/w2)
        return samples, cheby_weights(samples)
    if mode == 'christoffel' :
        polys = [ss.legendre(i)*np.sqrt((2*i + 1)) for i in range(multis.maxDegree+1)]
        polys = [(p*p).integ() for p in polys]
        d = multis.dim
        m = multis.cardinality
        samples = np.random.uniform(low=-1, high=1, size=(d, n))
        weights = np.zeros((n,))
        for j in range(n) :
            idx = multis[np.random.randint(low=0, high=m)].asList()
            for i in range(d) :
                samples[i,j] = bisection(polys[idx[i]], samples[i,j])
            weights[j] = m/np.sum([np.prod([polys[k](samples[l,j])**2 for l,k in enumerate(multis[q].asList())]) for q in range(multis.size())])
        return samples, weights

def get_sample_points_and_weights_deterministic(multis, mode, det_mode, n='wls') :
    points = {'cheby_det' : chebychev_1d, 'leja' : leja_1d, 'leggaus' : leggaus}[mode]

    #TODO think through weights for leja or leggaus below!
    samples = None
    if mode == 'shuffled':
        res = []
        for i in range(multis.dim) :
            base = points(n)
            randutil.rng.shuffle(base)
            res.append(base)
        samples = np.array(res)
    elif mode == 'tp_light':
        res = list(itertools.product(points(int(np.ceil(n**(1/multis.dim)))), repeat=multis.dim))
        samples = np.array(randutil.rng.choice(res, size=n, replace=False)).T
    elif mode == 'sparse_grid' :
        samples = n.getSparseGrid(lambda kmax : np.array(points(kmax)))
    else : assert(False)

    #require.equal(samples.shape, 'samples.shape', (multis.dim, n), '(multis.dim, n)')
    return samples, cheby_weights(samples)

def get_sample_points_and_weights(multis, mode, det_mode=None, n='wls') :
    assert(mode in ['uni', 'cheby', 'cheby_ss', 'christoffel', 'cheby_det', 'leja', 'leggaus'])
    if n == 'ip'  : n = multis.size()
    if n == 'wls' : n = 10 * multis.size() * int(np.log(multis.size()))
    if n == 'ls'  : n = multis.size()**2

    if mode in ['uni', 'cheby', 'cheby_ss', 'christoffel'] :
        return get_sample_points_and_weights_random(multis, mode, n)
    assert(False)


if __name__ == '__main__' :
    import MultiIndex as mi
    import require

    m = mi.TotalDegreeSet(dim=1, order=9)

    for mode in ['christoffel'] :
        s, w = get_sample_points_and_weights(m, mode)
        print(s.shape, 0 if w is None else w.shape)
        require.equal(s.shape[0], m.dim, 's.shape[0]', 'm.dim')
        if w is not None : require.equal(w.shape[0], s.shape[1], 'w.shape[0]', 's.shape[1]')
        #s, w = get_sample_points_and_weights('sparse_grid', m, points, n=m)
        #require.equal(s.shape[0], 's.shape[0]', m.dim, 'm.dim')