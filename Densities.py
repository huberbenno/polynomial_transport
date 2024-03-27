import numpy as np
from numpy.polynomial.legendre import legvander
import copy

import require, basis, randutil
import Database as db
import Forward as fw


class TargetDensity :

    def __init__(self, dim, name) :
        self.dim  = dim
        self.name = name
        self.norm = None
        self.norm_lebesgue = None

    def eval(self, x) :
        assert(isinstance(x, np.ndarray))
        assert(len(x.shape) <= 2)
        require.equal(x.shape[0], self.dim, 'x.shape[0]', 'self.dim')
        n = 1 if len(x.shape) == 1 else x.shape[1]
        y = self.__eval__(x)
        assert(isinstance(y, np.ndarray))
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

        N, s, s2 = 0, 0, 0
        while N < max_N :
            n = max(min_N, int(0.2 * N))
            N += n
            x = randutil.points(self.dim, n)
            for i in range(n) :
                fx = self.eval(np.expand_dims(x[:,i],-1))
                s  += fx
                s2 += fx**2
            if np.sqrt(s2 - s**2 / N) / s < accurc : break
        self.norm = s/N
        self.norm_lebesgue = 2**(self.dim) * self.norm

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()


class Gaussian1D(TargetDensity) :

    def __init__(self, *, mean=None, cova=None, save=False) :
        TargetDensity.__init__(self, 1, 'gaussian')
        self.mean = mean
        self.cova = cova
        self.norm = np.sqrt(2*np.pi * cova)
        if save :
            self.dbo, _ = db.GaussianDBO.get_or_create(
                dim=1, mean=db.to_string(mean), cova=db.to_string(cova))

    def eval(self, x) :
        if len(x.shape) > 1 : x = np.squeeze(x)
        return np.exp(-.5 * (x - self.mean)**2 / self.cova) / self.norm


class Gaussian(TargetDensity) :

    def __init__(self, *, mean, cova, save=False) :
        assert(len(mean.shape) == 1 or mean.shape[1] == 1)
        self.mean = np.array(mean)
        self.cova = np.array(cova)
        if len(self.mean.shape) == 1 : self.mean = np.expand_dims(self.mean, axis=1)
        TargetDensity.__init__(self, mean.shape[0], 'gaussian')
        require.equal(self.dim, cova.shape[0], 'self.dim', 'cova.shape[0]')
        require.equal(self.dim, cova.shape[1], 'self.dim', 'cova.shape[1]')

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
        assert(exponent >= 1)
        self.mean = np.array(mean)
        if len(self.mean.shape) == 1 : self.mean = np.expand_dims(self.mean, axis=1)
        TargetDensity.__init__(self, self.mean.shape[0], 'dyinggaussian')
        self.weights = np.expand_dims(np.array([(j+2)**(-exponent) for j in range(self.dim)]),axis=1)
        if save :
            self.dbo, _ = db.GaussianDBO.get_or_create(
                dim=self.dim, mean=db.to_string(self.mean),
                cova=db.to_string(np.diag([1/w for w in self.weights])), diag=True)

    def eval(self, x) :
        diff = self.mean - 100*np.multiply(x, self.weights)
        return np.exp(-.5 * np.sum(diff**2,axis=0))


class GaussianMixture(TargetDensity) :

    def __init__(self, *, dim, arglist, save=False) :
        TargetDensity.__init__(self, dim, 'gaussianmm')
        assert(len(arglist) > 1 and len(arglist) < 5)
        self.n = len(arglist)
        self.gaussians = [Gaussian(mean=arg['mean'], cova=arg['cova'], save=save) for arg in arglist]
        if save :
            glist = [g.dbo.id for g in self.gaussians] + [None]*(5 - len(self.gaussians))
            self.dbo, _ = db.GaussianMmDBO.get_or_create(
                dim=self.dim, gauss1=glist[0], gauss2=glist[1], gauss3=glist[2], gauss4=glist[3], gauss5=glist[4])

    def __eval__(self, x) :
        return np.sum([g.eval(x) for g in self.gaussians], axis=0) / self.n

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()
        for g in self.gaussians :
            g.deleteDbo()


class GaussianPosterior(TargetDensity) :

    def __init__(self, *, forwd, truep, noise, gauss=None, save=False) :
        require.equal(forwd.dim, len(truep), 'forwd.dim', 'len(truep)')
        require.notNone(forwd.xmeas, 'forwd.xmeas')
        TargetDensity.__init__(self, forwd.dim, 'posterior')
        self.forwd = forwd
        self.truep = truep
        self.noise = noise
        self.gauss = gauss
        if self.gauss is None :
            mean = forwd.eval(truep)
            self.gauss = Gaussian(mean=mean + noise*randutil.rng.normal(size=mean.shape),
                                  cova=noise*np.eye(len(mean)), save=save)
        #self.norm = np.sqrt((2*np.pi)**forwd.dim * np.linalg.det(np.linalg.inv(np.dot(forwd.M.T,np.dot(self.gauss.invc, forwd.M)))))

        if save :
            self.dbo, _ = db.GaussianPosteriorDBO.get_or_create(
                forwd=self.forwd.dbo.id, gauss=self.gauss.dbo.id, truep=db.to_string(truep), noise=noise, xmeas=db.to_string(forwd.xmeas))

    def eval(self, x):
        return self.gauss.eval(self.forwd.eval(x))

    @classmethod
    def fromConfig(self, *, fwd, noise) :
        dbos = db.GaussianPosteriorDBO.select().where(db.GaussianPosteriorDBO.forwd_id == fwd.dbo.id,
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


class Kdiff(TargetDensity) :

    def __init__(self, *, dim, k) :
        TargetDensity.__init__(self, dim, 'kdiff')
        self.k = k
        self.norm = 2 * (1.1 - 1/(k+1))

    def eval(self, x) :
        return (1.1 - np.prod(x, axis=0)**self.k)/self.norm


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

    def eval(self, x) :
        x = self.scale * np.dot(self.rotation, x - self.centr[:, np.newaxis])
        return np.exp(-(self.a - x[0])**2 - self.b * (x[1] - x[0]**2)**2)


class Circle(TargetDensity) :
    def __init__(self, *, c, r, w) :
        TargetDensity.__init__(self, 2, 'circle')
        self.c = c
        self.r = r
        self.w = w
        #self.dbo, _ = db.CircleDBO.get_or_create(c=db.to_string(c), r=r, w=w)

    def eval(self, x) :
        return np.exp(-(abs((x[0,:]-self.c[0])**2 + (x[1,:]-self.c[1])**2 - self.r))/self.w)


class Hat(TargetDensity) :
    def __init__(self, *, c=[-.5,.3], m=1, x=[-.7,-.3], y=[.2,.6], theta=0, scale=1) :
        assert(x[0] <= c[0] and c[0] <= x[1] and y[0] <= c[1] and c[1] <= y[1])
        TargetDensity.__init__(self, 2, 'Hats')
        self.arcsins1 = np.array([m/(c[0]-x[0]), m/(c[1]-y[0]), m/(x[1]-c[0]), m/(y[1]-c[1])])
        self.c = np.array(c)
        self.x = x
        self.y = y
        self.rotation = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.scale = scale

    def eval(self, x) :
        x = self.scale * np.dot(self.rotation, x)
        res = np.zeros((x.shape[1]))
        ind = (x[0,:] >= self.x[0]) & (x[0,:] <= self.x[1]) & (x[1,:] >= self.y[0]) & (x[1,:] <= self.y[1])
        if ind.any() :
            res[ind] = np.vstack((self.arcsins1[0]*np.abs(self.x[0]-x[0,ind]), self.arcsins1[1]*np.abs(self.y[0]-x[1,ind]), self.arcsins1[2]*np.abs(self.x[1]-x[0,ind]), self.arcsins1[3]*np.abs(self.y[1]-x[1,ind]))).min(axis=0)
        return res


class MultimodalDensity(TargetDensity) :
    def __init__(self, *, densities, weights) :
        assert(len(densities) == len(weights))
        TargetDensity.__init__(self, 2, 'MultimodalDensity')
        self.densities = densities
        self.weights = weights

    def eval(self, x) :
        return np.sum([w*d.eval(x) for w, d in zip(self.weights,self.densities)], axis=0)


class Uniform(TargetDensity) :
    def __init__(self, *, dim=2, c=1) :
        TargetDensity.__init__(self, dim, 'Uniform')
        self.c = c

    def eval(self, x) :
        return np.ones(x.shape[1]) * self.c / 2**self.dim


class IntermediateDensity(TargetDensity) :
    def __init__(self, multitree) :
        TargetDensity.__init__(self, 2, 'IntermediateDensity')
        self.multitree = multitree

    def eval(self, x) :
        m = self.multitree.maxOrders[0]
        r = 0
        p_x = legvander(np.array([x[0]]), m+2)[0].T
        for i in range(p_x.shape[0]) :
            p_x[i] *= np.sqrt((2*i + 1)/2)
        for n in self.multitree[1] :
            v = [c.val for c in n.children]
            j = [c.idx for c in n.children]
            r += np.sum([v[i]*p_x[j[i]] for i in range(len(v))], axis=0)**2
        return r


def generate_densities(save) :
    densities = []
    for d in [1, 3, 5] :
        arglist = [{'mean' : randutil.points(d,1), 'cova' : randutil.covarm(d)},
                   {'mean' : randutil.points(d,1), 'cova' : randutil.covarm(d)}]
        densities += [GaussianMixture(dim=d, arglist=arglist, save=save)]
        #densities += [DyingGaussian(mean=randutil.points(d,1), save=save)]
    for d in randutil.rng.integers(low=1, high=32, size=(3,)) :
        f = fw.Convolution(basis=basis.hats, dim=d, alpha=1, xmeas=randutil.points(10), save=save)
        densities += [GaussianPosterior(forwd=f, truep=randutil.points(d), noise=randutil.rng.uniform()/10)]
        #TODO  Kdiff
        #densities += [Rosenbrock(a=.2, b=8, theta=-2.2/10*np.pi, centr=np.array([.5,-.5]), scale=1.1)]
    densities += [Circle(c=randutil.points(2), r=.4, w=.2)]
    densities += [Hat()]
    return densities

if __name__ == '__main__' :
    import logutil, require

    logutil.print_start('Testing Density Module...', end='\n')

    for save in [True, False] :

        for t in generate_densities(save) :
            require.equal(t.eval(randutil.points(t.dim, 1)).shape, (1,), 'shape return value of eval single point', 'expected')
            require.equal(t.eval(randutil.points(t.dim,10)).shape, (10,), 'shape return value of eval multiple points', 'expected')
            t.deleteDbo()

    logutil.print_done()
