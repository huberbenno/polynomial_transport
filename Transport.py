import numpy as np

from polynomial_transport import util
from . import MultiIndex as mi


class TransportMap :

    def __init__(self, surrogate) :
        self.surrogate = surrogate
        self.multitree = mi.MultiIndexTree(surrogate)
        self.d = surrogate.multis.dim

    def eval(self, x) :
        """
        x : np.array with shape=(d,)
        """
        x = np.atleast_1d(np.squeeze(np.array(x)))
        assert len(x.shape) == 1 and len(x) == self.d

        s = self.surrogate.norm_lebesgue
        S = np.zeros((self.d,))
        for i in range(self.d) :
            S[i], s = self.eval_i(i, x[i], s)
        return S

    def eval_i(self, i, x, s) :
        r = 0
        ss = 0
        L, I = util.legendre.get_integrated_products(self.multitree.maxOrders[i] + 1, x)

        for n in self.multitree[i+1] :
            v = [c.val for c in n.children]
            j = [c.idx for c in n.children]

            r += np.dot(v, np.dot(I[np.ix_(j,j)], v))

            if i < self.d-1 :
                n.val = np.dot(v, L[j])
                ss += n.val**2

        return 2*r/s - 1, ss

    def inveval(self, y) :
        midpoint = lambda interval : (interval[1] + interval[0])/2

        x = np.zeros((self.d,))
        s = self.surrogate.norm_lebesgue

        for i in range(self.d) :
            interval = [-1,1]
            ss = 0

            for _ in range(30) : # 2^-14 < 1e-4
                candidate_i, ss = self.eval_i(i, x[i], s)
                if candidate_i > y[i] :
                    interval = [interval[0],  x[i]]
                else :
                    interval = [x[i], interval[1]]
                x[i] = midpoint(interval)
            s = ss

        return x

    def samples(self, n, p_uni=None) :
        if p_uni is None :
            p_uni = util.random.points(2, n)
        p_tar = np.zeros(p_uni.shape)
        for j in range(p_uni.shape[1]) :
            p_tar[:,j] = self.inveval(p_uni[:,j])
        return p_uni, p_tar

    def grid(self, xs=(-1, -.9, -.8, -.4, 0, .4, .8, .9, 1), ns=(100, 50, 20, 20, 20, 20, 50, 100)) :
        assert self.d == 2
        lines = []
        k = np.sum(ns) + 1
        l = np.concatenate([np.linspace(xs[i], xs[i+1], ns[i], endpoint=False) for i in range(len(ns))] + [[1]])
        for i in np.linspace(-1,1,11, endpoint=True) :
            lines.append(np.array([[i,i], [-1,1]]))
            if i in [-1, 1] :
                lines.append(np.array([[-1,1], [i,i]]))
            else :
                ll = np.ones((2,k))*i
                ll[0,:] = l
                lines.append(ll)

        lines_t = []
        for l in lines :
            lines_t.append(np.array([np.array(self.inveval(x)) for x in l.T]).T)
        return lines, lines_t
