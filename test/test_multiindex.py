import numpy as np

import util

import MultiIndex as mi

import setup


def test_multiindex() :

    util.log.print_start('Testing Multiindex Module...', end='\n')

    setup.database()

    for save in [True, False] :

        m = mi.AnisotropicSet(weights=np.log([1/.6]), cardinality=5, save=save, verbose=1)
        assert m.cardinality == 5
        if save : m.deleteDbo()

        m = mi.AnisotropicSet(weights=np.log([1/.6, 1/.4]), cardinality=27, save=save, verbose=1)
        m.print()
        assert m.cardinality == 27
        m.deleteDbo()

        m = mi.AnisotropicSet(weights=np.log([1/.6, 1/.4, 1/.1, 1/.01]), cardinality=31, save=save, verbose=1)
        assert m.cardinality == 31
        m.deleteDbo()

        m = mi.TensorProductSet(dim=1, order=5, save=save)
        assert m.cardinality == 6
        if save : m.deleteDbo()

        m = mi.TensorProductSet(dim=2, order=5, save=save)
        assert m.cardinality == 36
        m.deleteDbo()

        m = mi.TensorProductSet(dim=3, order=5, save=save)
        assert m.cardinality == 216
        m.deleteDbo()

        m = mi.TotalDegreeSet(dim=1, order=5, save=save)
        assert m.cardinality == 6
        if save : m.deleteDbo()

        m = mi.TotalDegreeSet(dim=2, order=5, save=save)
        assert m.cardinality == 21
        m.deleteDbo()

        m = mi.TotalDegreeSet(dim=3, order=5, save=save)
        assert m.cardinality == 56
        m.deleteDbo()

    util.log.print_done()
