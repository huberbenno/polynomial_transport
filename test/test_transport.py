
import util

import Densities as de
import MultiIndex as mi
import Surrogates as su
import Transport as tm

import setup

def test_transport() :

    util.log.print_start('Testing Transport Module...', end='\n')
    util.log.print_indent('d = 1')

    setup.database()

    t = de.Gaussian(mean=util.random.points(1, 1), cova=util.random.covarm(1))
    m = mi.TensorProductSet(dim=1, order=5)
    s = su.Legendre(multis=m, target=t)

    tt = tm.TransportMap(s)
    tt.test()

    s.deleteDbo()
    m.deleteDbo()
    t.deleteDbo()

    util.log.print_indent('d = 2')

    t = de.Gaussian(mean=util.random.points(2, 1), cova=util.random.covarm(2))
    #t = de.Rosenbrock(a=.15, b=10)
    m = mi.TotalDegreeSet(dim=2, order=3)
    s = su.Legendre(multis=m, target=t)

    tt = tm.TransportMap(s)
    tt.test()

    s.deleteDbo()
    m.deleteDbo()
    t.deleteDbo()

    util.log.print_done()
