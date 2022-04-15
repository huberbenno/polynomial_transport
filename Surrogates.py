import numpy as np
import scipy.special as ss
import time

import MultiIndex as mi
import Database as db

import lejautil, legendreutil, require

def get_sample_points_and_weights(method, multis) :
    d = multis.dim
    m = multis.size()
    assert(m > 0)
    samples = None
    weights = None
    if method == 'ls' :
        nevals = 10 * m**2
        samples = np.random.uniform(low=-1, high=1, size=(d, nevals))
    elif method == 'wls' :
        nevals = 10* m * int(np.log(m))
        samples = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2, size=(d, nevals)))
        weights = np.pi/2 * np.prod(np.sqrt(1-samples**2), axis=0)
    elif method == 'ip_leja' :
        samples = lejautil.leja_points(multis)
    elif method == 'ip_leggaus' :
        samples, weights = lejautil.leggaus_points(multis)
    elif method == 'wls_leja' :
        k = 1
        if multis.dim == 1 :
            k = int(1.5 * multis.maxDegree)
        else :
            while ss.binom(k+1, multis.dim-1) <= 1.5 * multis.cardinality :
                k += 1
        m = mi.TotalDegreeSet(dim=multis.dim, order=k)
        samples = lejautil.leja_points(m)
        m.deleteDbo()
    elif method == 'wls_leggaus' :
        k = 1
        if multis.dim == 1 :
            k = int(1.5 * multis.maxDegree)
        else :
            while ss.binom(k+1, multis.dim-1) <= 1.5 * multis.cardinality :
                k += 1
        m = mi.TotalDegreeSet(dim=multis.dim, order=k)
        samples, weights = lejautil.leggaus_points(m)
        m.deleteDbo()
    else :
        assert(False)
    return samples, weights

class Legendre :

    def __init__(self, *, multis, target, method, save=False, verbose=False) :
        assert(multis.dim == target.dim)
        self.multis = multis
        self.dbo, isnew = db.SurrogateDBO.get_or_create(
            target=target.name, target_id=target.dbo.id, multis=multis.name, multis_id=multis.dbo.id, method=method)
        #print(self.dbo.coeffs, isnew, target.name, target.dbo.id, multis.name, multis.dbo.id, method)
        if not isnew and self.dbo.coeffs is not None :
            self.coeffs = db.fr_string(self.dbo.coeffs)
        else :
            if verbose : print('Surrogate...', end=' ')
            start = time.process_time()

            points, weights = get_sample_points_and_weights(method, multis)
            rhs = np.squeeze(target.evalSqrt(points))
            lhs = legendreutil.evaluate_basis(points, multis)

            #print(points.shape, weights.shape, rhs.shape, lhs.shape)
            if weights is not None:
                for i in range(lhs.shape[0]) :
                    lhs[i,:] *= weights[i]
                    rhs[i] *= weights[i]

            self.coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

            self.dbo.ctime = time.process_time() - start
            self.dbo.nevals = points.shape[1]
            self.dbo.coeffs = db.to_string(self.coeffs)
            self.dbo.save()
            if verbose : print('Done')

        require.equal(self.coeffs.shape, 'coeffs.shape', (multis.size(),), 'len(multis)')
        self.norm = np.sum(self.coeffs**2)
        self.m = len(self.coeffs)

    def evalSqrt(self, x) :
        if isinstance(x, list) : x = np.array(x)
        elif not hasattr(x, '__len__') : x = np.array([[x]])
        if x.ndim == 1 : x = np.expand_dims(x, axis=0)
        basis = legendreutil.evaluate_basis(x, self.multis)
        return np.dot(basis, self.coeffs)

    def eval(self, x) :
        return self.evalSqrt(x)**2

    def deleteDbo(self) :
        self.dbo.delete_instance()

    #TODO move legendreutil.evaluate_basis here

if __name__ == '__main__' :
    import Densities as de
    import randutil as rd

    t = de.Gaussian(mean=rd.points(1,1), cova=rd.covarm(1))
    p = rd.points(1,1)
    print(t.eval(p))
    m = mi.TensorProductSet(dim=1, order=15)

    for methd in ['ls', 'wls', 'ip_leja', 'ip_leggaus', 'wls_leja', 'wls_leggaus'] :
        s = Legendre(multis=m, target=t, method=methd)
        print(methd.ljust(11), s.eval(p))
        s.deleteDbo()

    t.deleteDbo()
    m.deleteDbo()

    print()

    t = de.Gaussian(mean=rd.points(2,1), cova=rd.covarm(2))
    p = rd.points(2,1)
    print(t.eval(p))
    m = mi.TensorProductSet(dim=2, order=6)

    for methd in ['ls', 'wls', 'ip_leja', 'ip_leggaus', 'wls_leja', 'wls_leggaus'] :
        s = Legendre(multis=m, target=t, method=methd)
        print(methd.ljust(11), s.eval(p))
        s.deleteDbo()

    t.deleteDbo()
    m.deleteDbo()