import numpy as np
import pymuqModeling_ as mm
import basis

class Convolution(mm.PyModPiece):

    def __init__(self, *, basis, d, alpha, x_val, n_quad=10):
        super(Convolution, self).__init__([d], [len(x_val)])

        self.basis = basis
        self.alpha = alpha
        self.x_val = x_val

        self.l = np.ceil(np.log2(d))
        self.n = int(2**self.l)

        x_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
        self.x_quad = np.concatenate([x_quad/self.n - 1 + (2*i+1)/self.n for i in range(self.n)])
        self.w_quad = np.concatenate([w_quad for i in range(self.n)])

    def EvaluateImpl(self, inputs):
        param = inputs[0]

        q_x = np.multiply([self.basis(xi, param, self.alpha) for xi in self.x_quad], self.w_quad)

        res = np.zeros((len(self.x_val),))
        for i in range(len(self.x_val)) :
            e_x = [np.exp(-10*(self.x_val[i] - xj)**2) for xj in self.x_quad]
            res[i] = np.dot(q_x, e_x) / self.n

        self.outputs = [res]

if __name__ == '__main__' :

    conv = Convolution(basis=basis.hats, d=5, alpha=1, x_val=np.linspace(-1,1,20))
    evls = conv.Evaluate([[.1,.2,.23,-.435,-.23]])
