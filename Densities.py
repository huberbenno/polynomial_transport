import numpy as np

import require, basis
import Database as db
import Forward as fw


class TargetDensity :

    def __init__(self, dim, name) :
        self.dim  = dim
        self.name = name

    def evalSqrt(self, x) :
        return np.sqrt(self.eval(x))

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
        require.equal(self.dim, 'self.dim', cova.shape[0], 'cova.shape[0]')
        require.equal(self.dim, 'self.dim', cova.shape[1], 'cova.shape[1]')
        self.invc = np.linalg.inv(cova)
        self.norm = np.sqrt((2*np.pi)**self.dim * np.linalg.det(cova))
        if save :
            self.dbo, _ = db.GaussianDBO.get_or_create(
                dim=self.dim, mean=db.to_string(mean), cova=db.to_string(cova),
                diag=np.all(cova == np.diag(np.diagonal(cova))))

    def eval(self, x) :
        diff = np.subtract(x, self.mean)
        return np.exp(-.5 * np.einsum('ij, ij -> j', diff, np.dot(self.invc, diff))) / self.norm


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
            glist =  [g.dbo.id for g in self.gaussians] + [None]*(5 - len(self.gaussians))
            self.dbo, _ = db.GaussianMmDBO.get_or_create(
                dim=self.dim, gauss1=glist[0], gauss2=glist[1], gauss3=glist[2], gauss4=glist[3], gauss5=glist[4])

    def eval(self, x) :
        return np.sum([g.eval(x) for g in self.gaussians], axis=0) / self.n

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()
        for g in self.gaussians :
            g.deleteDbo()


class GaussianPosterior(TargetDensity) :

    def __init__(self, *, forwd, truep, xeval, xmsrmt, noise, save=False) :
        TargetDensity.__init__(self, len(truep), 'posterior')
        self.gauss = Gaussian(mean=xmsrmt, cova=noise*np.eye(len(xmsrmt)), save=save)
        self.forwd = forwd
        self.xeval = xeval
        if save :
            self.dbo, _ = db.GaussianPosteriorDBO.get_or_create(
                forwd=self.forwd.dbo.id, gauss=self.gauss.dbo.id, truep=db.to_string(truep),
                xeval=db.to_string(xeval))

    def eval(self, x):
        x = self.forwd.eval(x, xeval=self.xeval)
        return self.gauss.eval(x)

    @classmethod
    def fromId(self,id) :
        dbo = db.GaussianPosteriorDBO.get_by_id(id)
        dboGauss = db.GaussianDBO.get_by_id(dbo.gauss_id)
        return GaussianPosterior(forwd=fw.Convolution.fromId(dbo.forwd_id), truep=db.fr_string(dbo.truep),
                                 xeval=db.fr_string(dbo.xeval), xmsrmt=db.fr_string(dboGauss.mean),
                                 noise=db.fr_string(dboGauss.cova)[0], save=True)

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


if __name__ == '__main__' :
    import randutil as rd
    save = False

    for d in [1, 2, 3, 4, 5, 10] :
        print('d = ', d)

        arglist = [{'mean' : rd.points(d,1), 'cova' : rd.covarm(d)},
                   {'mean' : rd.points(d,1), 'cova' : rd.covarm(d)}]
        t = GaussianMixture(dim=d, arglist=arglist, save=save)
        print('\t', t.eval(rd.points(d,1)).shape, t.eval(rd.points(d, 10)).shape)
        t.deleteDbo()

        t = DyingGaussian(mean=rd.points(d,1), save=save)
        print('\t', t.eval(rd.points(d,1)).shape, t.eval(rd.points(d, 10)).shape)
        t.deleteDbo()

        f = fw.Convolution(basis=basis.hats, dim=d, alpha=1, noise=.1, save=save)
        xe = np.linspace(-1,1,20)
        xm = f.eval(rd.points(d,1), xe)
        pts = rd.points(d,10)
        t1 = GaussianPosterior(forwd=f, truep=rd.points(d), xeval=xe, xmsrmt=xm, noise=.1, save=save)
        eval1 = t1.eval(pts)
        if save :
            t2 = GaussianPosterior.fromId(t1.dbo.id)
            eval2 = t2.eval(pts)
            require.close(eval1, eval2)
            t2.deleteDbo()
        t1.deleteDbo()
        f.deleteDbo()