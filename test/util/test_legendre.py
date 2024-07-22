import time
import numpy as np

from util import legendre, require, random

import MultiIndex as mi


def test_get_polys() :
    """ test orthonormality """
    polys = legendre.get_polys(20)
    for i in range(len(polys)) :
        antid = (polys[i] * polys[i]).integ()
        require.close(antid(1) - antid(-1), 1, atol=1e-3)  # supposed to be 1
        for j in range(i+1, len(polys)) :
            antid = (polys[i] * polys[j]).integ()
            require.close(antid(1) - antid(-1), 0, atol=1e-3)  # supposed to be 0


def test_get_integrated_products() :
    m = 20
    x = random.points(1)[0,0]

    res = legendre.get_integrated_products(m, x)[1]

    polys = legendre.get_polys(m)
    for i in range(m):
        for j in range(i, m):
            antid = np.polymul(polys[i], polys[j]).integ()
            val = antid(x) - antid(-1)
            require.close(val, res[i, j], atol=1e-3)
            require.close(val, res[j, i], atol=1e-3)


def test_evaluate_basis_equality():
    multis = mi.AnisotropicSet(weights=np.log([1/.6, 1/.4, 1/.3]), cardinality=7)

    for _ in range(100):
        x = random.points(multis.dim, 1)
        r_new = legendre.evaluate_basis(x, multis)
        r_old = legendre.evaluate_basis(x, multis, 'old')
        assert r_old.dtype == r_new.dtype
        require.close(r_old, r_new)


def test_evaluate_basis_runtime() :

    multis = mi.AnisotropicSet(weights=np.log([1/.6, 1/.4, 1/.3]), cardinality=7)

    start = time.process_time()
    for _ in range(100) :
        x = random.points(multis.dim, 1)
        r_new = legendre.evaluate_basis(x, multis)
    time_old = time.process_time() - start

    start = time.process_time()
    for _ in range(100) :
        x = random.points(multis.dim, 1)
        r_old = legendre.evaluate_basis(x, multis, 'old')
    time_new = time.process_time() - start

    print(f'Runtime old : {time_old} / Runtime new : {time_new}')
