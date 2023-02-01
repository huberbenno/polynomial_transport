import numpy as np

import basis
import Database as db


class Convolution :

    def __init__(self, *, basis, dim, alpha, x_measurement=None, k_width=10, n_quad=10, save=False) :
        self.basis = basis
        self.dim   = dim
        self.alpha = alpha
        self.k_width = k_width
        self.n = int(2**np.ceil(np.log2(dim))) # next greater power of two

        x_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
        self.x_quad = np.concatenate([x_quad/self.n - 1 + (2*i+1)/self.n for i in range(self.n)])
        self.w_quad = np.concatenate([w_quad for i in range(self.n)])

        self.M = None
        if x_measurement is not None :
            self.M = np.zeros((len(x_measurement), dim))
            for j in range(dim) :
                q_x = np.array([w * self.basis(x, [0]*j + [1], self.alpha) for x,w in zip(self.x_quad, self.w_quad)])

                for i in range(len(x_measurement)) :
                    e_x = np.array([np.exp(-self.k_width*(x_measurement[i] - xj)**2) for xj in self.x_quad])
                    self.M[i, j] = np.dot(q_x.T, e_x) / self.n

        if save :
            self.dbo, isnew = db.ForwardDBO.get_or_create(
                dim=dim, basis=basis.__name__, alpha=alpha, nquad=n_quad)


    def eval(self, p, x_eval=None) :
        assert(self.M is not None or x_eval is not None)
        assert(p.shape[0] == self.dim)

        if x_eval is None :
            return np.dot(self.M, p)

        assert(len(p.shape) == 1 or p.shape[1] == 1)
        q_x = np.array([w * self.basis(x, p, self.alpha) for x,w in zip(self.x_quad, self.w_quad)])

        res = np.zeros((len(x_eval),1))
        for i in range(len(x_eval)) :
            e_x = np.array([np.exp(-self.k_width*(x_eval[i] - xj)**2) for xj in self.x_quad])
            res[i] = np.dot(q_x.T, e_x) / self.n

        return res

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()

    @classmethod
    def fromId(self, id) :
        dbo = db.ForwardDBO.get_by_id(id)
        return Convolution(basis=getattr(basis, dbo.basis), dim=dbo.dim, alpha=dbo.alpha,
                           n_quad=dbo.nquad, save=True)

if __name__ == '__main__' :
    import randutil, require

    save = False
    dim = 13
    x = np.linspace(-1,1,20)
    p = randutil.points(dim)

    f = Convolution(basis=basis.hats, dim=dim, alpha=1, x_measurement=x, save=save)

    res1 = f.eval(p)
    res2 = f.eval(p, x_eval=x)
    for i in range(len(res1)) :
        require.close(res1[i], res2[i])

    if save :
        f2 = Convolution.fromId(f.dbo.id)
    f.deleteDbo()