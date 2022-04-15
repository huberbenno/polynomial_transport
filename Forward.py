import numpy as np

import basis
import Database as db


class Convolution :

    def __init__(self, *, basis, dim, alpha, noise, n_quad=10) :
        self.basis = basis
        self.alpha = alpha
        self.n = int(2**np.ceil(np.log2(dim))) # next greater power of two

        x_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
        self.x_quad = np.concatenate([x_quad/self.n - 1 + (2*i+1)/self.n for i in range(self.n)])
        self.w_quad = np.concatenate([w_quad for i in range(self.n)])
        self.weights = [2**(-alpha*np.ceil(np.log2(i+1))) for i in range(1,dim+1)]

        self.dbo, isnew = db.ForwardDBO.get_or_create(
            dim=dim, basis=basis.__name__, alpha=alpha, nquad=n_quad, noise=noise)

    def eval(self, p, xeval) :
        assert(p.shape[0] == self.dbo.dim)
        n = 1 if len(p.shape) == 1 else p.shape[1]
        #if len(p.shape) > 1 : p = np.squeeze(p)
        #print(p.shape, self.basis(self.x_quad[0], p, self.alpha).shape)
        q_x = np.array([w * self.basis(x, p, self.alpha) for x,w in zip(self.x_quad, self.w_quad)])

        res = np.zeros((len(xeval),n))
        for i in range(len(xeval)) :
            e_x = np.array([np.exp(-10*(xeval[i] - xj)**2) for xj in self.x_quad])
            #print(q_x.T.shape, e_x.shape, np.dot(q_x.T, e_x).shape)
            res[i] = np.dot(q_x.T, e_x) / self.n

        return res

    def deleteDbo(self) :
        self.dbo.delete_instance()

if __name__ == '__main__' :
    import randutil as rd
    dim = 5
    f = Convolution(basis=basis.hats, dim=dim, alpha=1, noise=.1)
    x = np.linspace(-1,1,20)
    print(f.eval(rd.points(dim), xeval=x))
    print(f.eval(rd.points(dim,10), xeval=x))
    f.deleteDbo()