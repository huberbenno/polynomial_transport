import numpy as np

import util

import Densities as de
import MultiIndex as mi
import Surrogates as su
import Transport as tm

import setup

def test_transport() :

    util.log.print_start('Testing Transport Module...', end='\n')

    setup.database()

    for d in [1,2,3] :
        util.log.print_indent(f'd = {d}')

        t = de.Gaussian(mean=util.random.points(d, 1), cova=util.random.covarm(d))
        m = mi.TensorProductSet(dim=d, order=2)
        s = su.Legendre(multis=m, target=t)

        tt = tm.TransportMap(s)

        print(' - testing domain boundaries ...')
        y = tt.eval([1]*tt.d)
        for yi in y : util.require.close(yi, 1)
        y = tt.eval([-1]*tt.d)
        for yi in y : util.require.close(yi, -1, atol=1e-4)

        print(' - testing inverse ... ', end='')
        for i in range(5) :
            print(i, end=' ')
            x = util.random.points(d, 0)
            y = tt.eval(x)
            util.require.close(tt.inveval(y), x, atol=1e-4)
        print()

        print(' - testing determinant ...', end='')
        delta = .00001
        for i in range(3) :
            print(i, end=' ')
            x = util.random.points(tt.d, 0)
            det_S = np.prod((tt.eval(x+delta) - tt.eval(x-delta)) / (2*delta))*2**(-d)
            # huge atol here because finite difference approximation is not very stable, in particular for higher d
            util.require.close(det_S, tt.surrogate.evalNrmd(np.expand_dims(x, axis=1))[0], atol=.5)
        print()

        s.deleteDbo()
        m.deleteDbo()
        t.deleteDbo()

    util.log.print_done()
