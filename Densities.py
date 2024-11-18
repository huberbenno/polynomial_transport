import numpy as np
from numpy.polynomial.legendre import legvander
import copy

import util
import Database as db
import Forward as fw

#TODO allow to provide a function handle


class TargetDensity :

    def __init__(self, dim, name) :
        self.dim  = dim
        self.name = name
        self.norm = None
        self.norm_lebesgue = None

    def __eval__(self, x):
        raise NotImplementedError('This method has to be implemented by subclasses!')

    def eval(self, x) :
        x = util.points.ensure_shape(x, self.dim)
        y = self.__eval__(x)
        assert isinstance(y, np.ndarray)
        return y

    def evalSqrt(self, x) :
        return np.sqrt(self.eval(x))

    def evalNrmd(self, x) :
        if self.norm is None : self.computeNorm()
        return self.eval(x)/self.norm_lebesgue

    def evalSqrtNrmd(self, x) :
        if self.norm is None : self.computeNorm()
        return self.evalSqrt(x) / np.sqrt(self.norm_lebesgue)

    def computeNorm(self, accurc=.01, min_N=100, max_N=1e5) :
        if self.norm is not None and self.norm_lebesgue is not None : return
        #f = lambda x : self.eval(np.expand_dims(x,-1))
        #res = adaptive_smolyak(f, tol=accurc, maxit=max_n, smax=self.dim)
        #self.norm_lebesgue = res[0][0]
        #self.norm = 2**(self.dim) * self.norm_lebesgue

        print('WARNING: Computing an MC approximation of the norm - this will likely be either expensive or inaccurate!')
        N, s, s2 = 0, 0, 0
        while N < max_N :
            n = max(min_N, int(0.2 * N))
            N += n
            x = util.random.points(self.dim, n)
            for i in range(n) :
                fx = self.eval(np.expand_dims(x[:,i],-1))
                s  += fx
                s2 += fx**2
            if np.sqrt(s2 - s**2 / N) / s < accurc : break
        self.norm = s/N
        self.norm_lebesgue = 2 ** self.dim * self.norm

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()


class Uniform(TargetDensity) :
    def __init__(self, *, dim=2, c=1) :
        TargetDensity.__init__(self, dim, 'Uniform')
        self.c = c

    def __eval__(self, x) :
        return np.ones(x.shape[1]) * self.c / 2**self.dim


class Gaussian(TargetDensity) :

    def __init__(self, *, mean, cova, save=False) :
        assert len(mean.shape) == 1 or mean.shape[1] == 1
        self.mean = np.array(mean)
        self.cova = np.array(cova)
        if len(self.mean.shape) == 1 : self.mean = np.expand_dims(self.mean, axis=1)
        TargetDensity.__init__(self, mean.shape[0], 'gaussian')
        util.require.equal(self.dim, cova.shape[0], 'self.dim', 'cova.shape[0]')
        util.require.equal(self.dim, cova.shape[1], 'self.dim', 'cova.shape[1]')

        self.invc = np.linalg.inv(cova)
        self.norm = np.sqrt((2*np.pi)**self.dim * np.linalg.det(cova))

        if save :
            self.dbo, _ = db.GaussianDBO.get_or_create(
                dim=self.dim, mean=db.to_string(mean), cova=db.to_string(cova),
                diag=np.all(cova == np.diag(np.diagonal(cova))))

    def __eval__(self, x) :
        diff = np.subtract(x, self.mean)
        return np.exp(-.5 * np.einsum('ij, ij -> j', diff, np.dot(self.invc, diff)))


class DyingGaussian(TargetDensity) :

    def __init__(self, *, mean, exponent=2, save=False) :
        assert exponent >= 1
        self.mean = np.array(mean)
        if len(self.mean.shape) == 1 : self.mean = np.expand_dims(self.mean, axis=1)
        TargetDensity.__init__(self, self.mean.shape[0], 'dyinggaussian')
        self.weights = np.expand_dims(np.array([(j+2)**(-exponent) for j in range(self.dim)]),axis=1)
        if save :
            self.dbo, _ = db.GaussianDBO.get_or_create(
                dim=self.dim, mean=db.to_string(self.mean),
                cova=db.to_string(np.diag([1/w for w in self.weights])), diag=True)

    def __eval__(self, x) :
        diff = self.mean - 100*np.multiply(x, self.weights)
        return np.exp(-.5 * np.sum(diff**2, axis=0))


class GaussianPosterior(TargetDensity) :

    def __init__(self, *, forwd, truep, noise, gauss=None, save=False) :
        util.require.equal(forwd.dim, len(truep), 'forwd.dim', 'len(truep)')
        util.require.notNone(forwd.xmeas, 'forwd.xmeas')
        TargetDensity.__init__(self, forwd.dim, 'posterior')
        self.forwd = forwd
        self.truep = truep
        self.noise = noise
        self.gauss = gauss
        if self.gauss is None :
            mean = forwd.eval(truep)
            self.gauss = Gaussian(mean=mean + noise * util.random.rng.normal(size=mean.shape),
                                  cova=noise*np.eye(len(mean)), save=save)
        #self.norm = np.sqrt((2*np.pi)**forwd.dim * np.linalg.det(np.linalg.inv(np.dot(forwd.M.T,np.dot(self.gauss.invc, forwd.M)))))

        if save :
            self.dbo, _ = db.GaussianPosteriorDBO.get_or_create(
                forwd=self.forwd.dbo.id, gauss=self.gauss.dbo.id, truep=db.to_string(truep), noise=noise, xmeas=db.to_string(forwd.xmeas))

    def __eval__(self, x):
        return self.gauss.eval(self.forwd.eval(x))

    @classmethod
    def fromConfig(cls, *, fwd, noise) :
        dbos = db.GaussianPosteriorDBO.select().where(db.GaussianPosteriorDBO.forwd == fwd.dbo.id,
                                                      db.GaussianPosteriorDBO.noise == noise,
                                                      db.GaussianPosteriorDBO.xmeas == db.to_string(fwd.xmeas))
        res = []
        for dbo in dbos :
            gauss = Gaussian(mean=db.fr_string(dbo.gauss.mean), cova=dbo.gauss.recover_cova(), save=True)
            res.append(GaussianPosterior(forwd=copy.deepcopy(fwd), truep=db.fr_string(dbo.truep), noise=dbo.noise, gauss=gauss, save=True))
        return res

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()
        if hasattr(self.forwd, 'dbo') : self.forwd.dbo.delete_instance()
        if hasattr(self.gauss, 'dbo') : self.gauss.dbo.delete_instance()


class Rosenbrock(TargetDensity) :

    def __init__(self, *, a=1, b=100, theta=0, centr=np.array([0,0]), scale=2, save=False) :
        TargetDensity.__init__(self, 2, 'rosenbrock')
        self.a = a
        self.b = b
        self.rotation = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.centr = centr
        self.scale = scale
        if save :
            self.dbo, _ = db.RosenbrockDBO.get_or_create(
                a=a, b=b, theta=theta, centr=db.to_string(centr), scale=scale)

    def __eval__(self, x) :
        x = self.scale * np.dot(self.rotation, x - self.centr[:, np.newaxis])
        return np.exp(-(self.a - x[0])**2 - self.b * (x[1] - x[0]**2)**2)


class Circle(TargetDensity) :
    def __init__(self, *, c, r, w) :
        TargetDensity.__init__(self, 2, 'circle')
        self.c = c
        self.r = r
        self.w = w

    def __eval__(self, x) :
        return np.exp(-(abs((x[0,:]-self.c[0])**2 + (x[1,:]-self.c[1])**2 - self.r))/self.w)


class Hat(TargetDensity) :
    def __init__(self, *, c=(-.5,.3), m=1, x=(-.7,-.3), y=(.2,.6), theta=0, scale=1) :
        assert x[0] <= c[0] <= x[1] and y[0] <= c[1] <= y[1]
        TargetDensity.__init__(self, 2, 'Hats')
        self.arcsins1 = np.array([m/(c[0]-x[0]), m/(c[1]-y[0]), m/(x[1]-c[0]), m/(y[1]-c[1])])
        self.c = np.array(c)
        self.x = x
        self.y = y
        self.rotation = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.scale = scale

    def __eval__(self, x) :
        x = np.dot(self.rotation, x - self.c[:, None]) + self.c[:, None]
        res = np.zeros((x.shape[1]))
        ind = (x[0,:] >= self.x[0]) & (x[0,:] <= self.x[1]) & (x[1,:] >= self.y[0]) & (x[1,:] <= self.y[1])
        if ind.any() :
            res[ind] = self.scale * np.vstack((self.arcsins1[0] * np.abs(self.x[0] - x[0, ind]),
                                  self.arcsins1[1] * np.abs(self.y[0] - x[1, ind]),
                                  self.arcsins1[2] * np.abs(self.x[1] - x[0, ind]),
                                  self.arcsins1[3] * np.abs(self.y[1] - x[1, ind]))).min(axis=0)
        return res


class MultimodalDensity(TargetDensity) :
    def __init__(self, *, densities, weights=None) :
        if weights is not None : assert len(densities) == len(weights)
        dim = densities[0].dim
        for d in densities : assert d.dim == dim
        TargetDensity.__init__(self, dim, 'MultimodalDensity')
        self.densities = densities
        self.weights = weights

    def __eval__(self, x) :
        if self.weights is None :
            return np.sum([d.eval(x) for d in self.densities], axis=0)
        else :
            return np.sum([w*d.eval(x) for w, d in zip(self.weights,self.densities)], axis=0)


