import numpy as np

import util

import Densities as de
import Forward as fw

import setup


def generate_densities(save) :
    densities = []

    for d in [1, util.random.rng.integers(low=2, high=8)] :
        gaussians = [de.Gaussian(mean=util.random.points(d, 1), cova=util.random.covarm(d), save=False),
                     de.Gaussian(mean=util.random.points(d, 1), cova=util.random.covarm(d), save=False)]
        densities += [de.MultimodalDensity(densities=gaussians)]

    d = util.random.rng.integers(low=4, high=8)
    densities += [de.DyingGaussian(mean=util.random.points(d,1), save=save)]

    d = util.random.rng.integers(low=8, high=32)
    f = fw.Convolution(basis=util.basis.hats, dim=d, alpha=1, xmeas=util.random.points(10), save=save)
    densities += [de.GaussianPosterior(forwd=f, truep=util.random.points(d), noise=util.random.rng.uniform() / 10)]

    delta = 10 * np.pi / 72
    thetas = [2 * np.pi + .3 - i * 2 / 3 * np.pi - delta for i in range(3)]
    cs = [.3 * np.array([np.cos(th), np.sin(th) - .2]) for th in [(6 + i * 8) / 12 * np.pi + delta for i in range(3)]]
    densities += [de.MultimodalDensity(
        densities=[de.Rosenbrock(a=.4, b=4, theta=t, centr=c, scale=3.5) for c, t in zip(cs, thetas)],
        weights=[1, 1, 1])]

    densities += [de.Circle(c=util.random.points(2), r=.4, w=.2)]
    densities += [de.Hat()]

    return densities


def test_densities() :

    util.log.print_start('Testing Density Module...', end='\n')

    setup.database()

    for save in [True, False] :

        for t in generate_densities(save) :
            util.require.equal(t.eval(util.random.points(t.dim, 1)).shape, (1,),
                               'shape return value of eval single point', 'expected')
            util.require.equal(t.eval(util.random.points(t.dim, 10)).shape, (10,),
                               'shape return value of eval multiple points', 'expected')
            t.deleteDbo()

    util.log.print_done()
