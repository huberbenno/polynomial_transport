import numpy as np

rng = np.random.default_rng()


def points(d, n=None) :
    if n is None :
        return rng.uniform(low=-1, high=1, size=(d,1))
    if n == 0 :
        return rng.uniform(low=-1, high=1, size=(d,))
    return rng.uniform(low=-1, high=1, size=(d,n))


def covarm(d) :
    evecs = []
    for i in range(d) :
        v = points(d, 1)
        for u in evecs :
            v -= np.dot(u.T, v) * u
        evecs.append(v/np.linalg.norm(v))
    evals = rng.uniform(low=.01, high=.1, size=(d,))
    cov = sum([evals[i] * np.outer(evecs[i], evecs[i]) for i in range(d)])
    return cov


def covard(d) :
    return np.diag(np.sort(rng.uniform(low=.01, high=.1, size=(d,)))[::-1])
