import numpy as np
import scipy.special as ss

def leja_points_1d(n) :
    r = [1]
    if n > 0 : r.append(-1)
    if n > 1 : r.append(0)
    for j in range(3, n+1) :
        if j % 2 == 0 :
            r.append(-r[j-1])
        else :
            r.append(np.sqrt((r[int((j+1)/2)] + 1) / 2))
    return np.array(r).reshape((len(r),1))

def leja_points(multiset) :
    r = leja_points_1d(multiset.maxDegree)
    p = np.zeros((multiset.dim, multiset.cardinality))
    for i in range(multiset.dim) :
        for j in range(multiset.cardinality) :
            p[i,j] = r[multiset[j][i]]
    return p

def leggaus_points(multiset) :
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

if __name__ == '__main__' :
    print(leja_points_1d(10))