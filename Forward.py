import numpy as np

from polynomial_transport import util
# from . import Database as db


class Forward :

    def __init__(self, *, dim) :
        self.dim = dim

    def eval(self, p, **kwargs) :
        if isinstance(p, list) : p = np.array(p)
        util.require.equal(p.shape[0], self.dim, 'p.shape[0]', 'self.dim')

        res = self.__eval__(p, **kwargs)

        assert isinstance(res, np.ndarray)
        assert len(res.shape) == len(p.shape)
        if len(p.shape) > 1 :
            assert res.shape[1] == p.shape[1]

        return res

    def __eval__(self, x):
        raise NotImplementedError('This method has to be implemented by subclasses!')


class Convolution(Forward) :
    """ Convolution of a function $f : [-1,1] -> [-1,1]$ expressed in some basis $(b_i)_{i=1}^d$
        (i.e. $f(x) = \\sum_{i=1}^d p_i b_i(x)$) with an exponential kernel $k(x) = exp(-wx)$ of given width $w$.
        $(f * k)(x) = \\int_{-1}^{1} f(t) k(x-t) dt$
    """

    def __init__(self, *, dim, basis, alpha, wkern=10, nquad=100, xmeas=None, save=False) :
        """ dim   (int > 0)  : number of basis functions
            basis (function) : any of the basis functions specified in the util.basis module
            alpha (float)    :
            wkern (float)    :
            nquad (int > 1)  :
            xmeas (np.array) : value(s) $x$ at which to compute $(f * k)(x)$
        """
        Forward.__init__(self, dim=dim)
        self.basis = basis
        self.alpha = alpha
        self.wkern = wkern
        self.n = int(2**np.ceil(np.log2(dim)))  # next greater power of two

        # Approximate the integral of the convolution by quadrature
        x_quad, w_quad = np.polynomial.legendre.leggauss(nquad)
        self.x_quad = np.concatenate([x_quad/self.n - 1 + (2*i+1)/self.n for i in range(self.n)])
        self.w_quad = np.concatenate([w_quad for i in range(self.n)])
        #TODO weights need to multiply interval length

        # If xmeas=$x$ is specified, construct matrix M specified by $M_ij = (b_i * k)(x_j)$.
        # Since convolution is a linear operation, such M allows to compute $(f * k)(x) = Mp$
        # for $f$ given as $f = \sum_{i=1}^d p_i b_i$.
        self.M = None
        if xmeas is not None :
            self.xmeas = np.squeeze(xmeas)
            assert self.xmeas.ndim == 1
            self.M = np.zeros((len(self.xmeas), dim))
            for j in range(dim) :
                # Vector of weighted evaluations of the j-th basis functions at the quadrature nodes
                q_x = np.array([w * self.basis(x, [0]*j + [1], self.alpha) for x,w in zip(self.x_quad, self.w_quad)]).T
                for i in range(len(self.xmeas)) :
                    e_x = np.array([np.exp(-self.wkern*(self.xmeas[i] - xj)**2) for xj in self.x_quad])
                    self.M[i, j] = np.dot(q_x, e_x) / self.n

        if save :
            self.dbo, isnew = db.ConvolutionDBO.get_or_create(
                dim=dim, basis=basis.__name__, alpha=alpha, nquad=nquad, wkern=wkern)

    def dimWeights(self) :
        return [np.log2(i+2) for i in range(self.dim)]

    def __eval__(self, p, xmeas=None) :
        assert self.M is not None or xmeas is not None
        assert p.shape[0] == self.dim

        if xmeas is None :
            return np.dot(self.M, p)

        assert len(p.shape) == 1 or p.shape[1] == 1
        q_x = np.array([w * self.basis(x, p, self.alpha) for x,w in zip(self.x_quad, self.w_quad)])

        res = np.zeros((len(xmeas),1))
        for i in range(len(xmeas)) :
            e_x = np.array([np.exp(-self.wkern*(xmeas[i] - xj)**2) for xj in self.x_quad])
            res[i] = np.dot(q_x.T, e_x) / self.n

        return res

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()

    @classmethod
    def fromId(cls, id) :
        dbo = db.ConvolutionDBO.get_by_id(id)
        return Convolution(basis=getattr(util.basis, dbo.basis), dim=dbo.dim, alpha=dbo.alpha,
                           nquad=dbo.nquad, save=True)
