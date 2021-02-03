import numpy as np

import legendreutil, testutil

from StructuredMultiIndexSet import StructuredMultiIndexSet

class TransportMap :

    def __init__(self, d, coeffs, multiset) :
        self.d = d
        self.m = len(coeffs)
        self.coeffs = coeffs
        self.norm = np.dot(coeffs, coeffs)
        self.multiset = multiset
        self.multilist = [multiset.IndexToMulti(i) for i in range(multiset.Size())]
        self.structure = StructuredMultiIndexSet(multiset)

        #self.test()

    def evaluate(self, x) :
        """
        x : np.array with shape=(d,)
        """

        S = np.zeros((len(x)))

        p = self.coeffs
        s = np.dot(p, p)

        for i in range(self.d) :
            L, I = legendreutil.get_integrated_products(max([m.GetValue(i) for m in self.multilist])+1, x[i])

            r = 0
            for group in self.structure.groups[i] :

                beta = [np.sum([p[k] for k in self.structure.clusters[i][j]]) for j in group]
                kappa = [self.multilist[self.structure.clusters[i][j][0]].GetValue(i) for j in group]

                r += np.sum([beta[j]*beta[k]*I[kappa[j], kappa[k]] for j in range(len(kappa)) for k in range(len(kappa))])
                #r += np.dot(beta, np.dot(I[kappa][:, kappa], beta))

            S[i] = 2*r/s - 1

            if i < self.d-1 :
                p = np.multiply(p, [L[self.multilist[j].GetValue(i)] for j in range(self.m)])
                s = np.sum([np.sum([p[k] for k in cluster])**2 for cluster in self.structure.clusters[i+1]])

        return S

    def inverse_evaluate(self, y) :
        midpoint = lambda interval : interval[0] + (interval[1] - interval[0])/2

        x = [0]*self.d
        candidate = self.evaluate(x)

        for i in range(self.d) :

            interval = [-1,1]

            for _ in range(30) : # 2^-14 < 1e-4

                if candidate[i] > y[i] :
                    interval = [interval[0],  x[i]]
                else :
                    interval = [x[i], interval[1]]

                x[i] = midpoint(interval)
                candidate = self.evaluate(x)

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
        y = self.evaluate(np.array([1]*self.d))
        for yi in y : testutil.assert_close(yi, 1)
        y = self.evaluate(np.array([-1]*self.d))
        for yi in y : testutil.assert_close(yi, -1)

        print(' - testing inverse ...')
        for i in range(3) :
            x = np.random.uniform(low=-1, high=1, size=(self.d,))
            y = self.evaluate(x)
            testutil.assert_close(x, self.inverse_evaluate(y))

        print(' - testing determinant ...')
        for i in range(3) :
            x = np.random.uniform(low=-1, high=1, size=(self.d,))
            delta = .001*np.ones(x.shape)
            differences = (self.evaluate(x+delta) - self.evaluate(x-delta))/2
            det_S = np.prod(differences)
            g_app = np.dot(legendreutil.evaluate_basis(np.expand_dims(x, axis=1), self.multiset), self.coeffs)[0]**2
            testutil.assert_close(g_app, det_S)
        print('All done!')

#if __name__ == '__main__' :

