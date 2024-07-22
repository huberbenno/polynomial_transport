import numpy as np

import util

import Densities as de
import MultiIndex as mi
import Surrogates as su

import setup


def test_legendre() :

    util.log.print_start('Testing Surrogate Module...', end='\n')
    util.log.print_indent('TARGET'.ljust(15) + 'D'.ljust(3) + 'MODE'.ljust(12) + 'DIST'.ljust(12)
                          + 'MULTIS'.ljust(7) + 'N_MC'.ljust(6) + ' ERROR')

    setup.database()

    for save in [True, False] :

        d = 3
        for t in [de.Gaussian(mean=util.random.points(2, 1), cova=util.random.covarm(2), save=save)] :
            m = mi.TotalDegreeSet(dim=t.dim, order=int(np.ceil(50**(1/t.dim))), save=save)

            for mode, dist in [('wls', 'cheby'), ('ip', 'leja'), ('ip', 'leggaus')] :
                s = su.Legendre(multis=m, target=t, mode=mode, dist=dist, save=save)
                s.eval(util.random.points(t.dim))
                e = s.computeError()
                if save : util.log.print_indent(t.name.ljust(15) + str(t.dim).ljust(3) + mode.ljust(12)
                                                + dist.ljust(12) + str(m.cardinality).ljust(7)
                                                + str(e.nevals).ljust(6) + ' {:.4f}'.format(e.hedist))
                e.deleteDbo()
                s.deleteDbo()
            t.deleteDbo()
            m.deleteDbo()

    util.log.print_done()
