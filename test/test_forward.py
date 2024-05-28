import util

from Forward import *

import setup


def test_convolution() :
    util.log.print_start('Testing Forward Module...', end='\n')

    setup.database()

    dim = util.random.rng.integers(low=1, high=10)
    util.log.print_indent(' Testing Convolution with dimension {}'.format(dim))

    x = np.linspace(-1, 1, 20)
    p = util.random.points(dim)

    for save in [True, False] :
        f = Convolution(basis=util.basis.hats, dim=dim, alpha=1, xmeas=x, save=save)

        res1 = f.eval(p)
        res2 = f.eval(p, xmeas=x)
        util.require.close(res1, res2)

        if save :
            f2 = Convolution.fromId(f.dbo.id)
        f.deleteDbo()

    util.log.print_done()
