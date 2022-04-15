import numpy as np

import legendreutil, require

import MultiIndex as mi

class TransportMap :

    def __init__(self, surrogate) :
        self.surrogate = surrogate
        self.multitree = mi.MultiIndexTree(surrogate)
        self.d = surrogate.multis.dim

    def eval(self, x) :
        """
        x : np.array with shape=(d,)
        """
        #x = np.squeeze(x)
        #print(x, x.size)
        #if x.size > 1 : assert(len(x) == self.d)

        S = np.zeros((self.d,))

        s = self.surrogate.norm

        for i in range(self.d) :
            L, I = legendreutil.get_integrated_products(self.multitree.maxOrders[i]+1, x[i])
            r = 0
            ss = 0

            for n in self.multitree[i+1] :

                v = [c.val for c in n.children]
                j = [c.idx for c in n.children]

                r += np.dot(v, np.dot(I[np.ix_(j,j)], v))

                if i < self.d-1 :
                    n.val = np.dot(v, L[j])
                    ss += n.val**2

            S[i] = 2*r/s - 1
            s = ss

        return S

    def inveval(self, y) :
        midpoint = lambda interval : interval[0] + (interval[1] - interval[0])/2

        x = [0]*self.d
        candidate = self.eval(x)

        for i in range(self.d) :

            interval = [-1,1]

            for _ in range(30) : # 2^-14 < 1e-4

                if candidate[i] > y[i] :
                    interval = [interval[0],  x[i]]
                else :
                    interval = [x[i], interval[1]]

                x[i] = midpoint(interval)
                candidate = self.eval(x)

        return x

    def density_sqrt(self, x) :
        return np.dot(legendreutil.evaluate_basis(np.expand_dims(x, axis=1), self.multiset), self.coeffs)[0]

    def density(self, x) :
        return self.density_sqrt(x)**2

    def det_dS(self, x) :
        return self.density(x) / self.norm

    def test(self) :

        print('Testing transport map functionality:')
        print(' - testing domain boundaries ...')
        y = self.eval(np.array([1]*self.d))
        for yi in y : require.close(yi, 1)
        y = self.eval(np.array([-1]*self.d))
        for yi in y : require.close(yi, -1)

        print(' - testing inverse ...')
        for i in range(3) :
            print(i)
            x = np.random.uniform(low=-1, high=1, size=(self.d,))
            y = self.eval(x)
            require.close(self.inveval(y), x)

        #print(' - testing determinant ...')
        #for i in range(3) :
        #    x = np.random.uniform(low=-1, high=1, size=(self.d,))
        #    delta = .0000001*np.ones(x.shape)
        #    differences = (self.eval(x+delta) - self.eval(x-delta))*20000000
        #    det_S = np.prod(differences)
        #    require.close(det_S, self.surrogate.eval(np.expand_dims(x, axis=1))[0])
        print('All done!')

if __name__ == '__main__' :
    import Densities as de
    import Surrogates as su
    import randutil as rd

    print('# ----- 1D -----')

    t = de.Gaussian(mean=rd.points(1,1), cova=rd.covarm(1))
    m = mi.TensorProductSet(dim=1, order=5)
    s = su.Legendre(multis=m, target=t, method='wls')

    tm = TransportMap(s)
    tm.test()

    s.deleteDbo()
    m.deleteDbo()
    t.deleteDbo()

    print('# ----- 2D -----')

    t = de.Gaussian(mean=rd.points(2,1), cova=rd.covarm(2))
    #t = de.Rosenbrock(a=.15, b=10)
    m = mi.TotalDegreeSet(dim=2, order=3)
    s = su.Legendre(multis=m, target=t, method='wls')

    tm = TransportMap(s)
    tm.test()

    s.deleteDbo()
    m.deleteDbo()
    t.deleteDbo()
