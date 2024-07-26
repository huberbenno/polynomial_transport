import os, sys, subprocess, datetime, itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import peewee as pw

sys.path.append('../')
import util
from Densities import *
from MultiIndex import *
from Surrogates import *
from Forward import *


QUIT = False


def create_surrogates(t, n_max=1000, verbose=1, save=True) :

    ns = np.logspace(np.log(3), np.log(n_max), num=10, endpoint=True, base=np.e, dtype=int)
    ns[0], ns[-1] = 3, n_max

    for n in np.unique(ns) :
        if QUIT : break
        print(f'd {t.dim} || alpha {t.forwd.alpha} || noise {t.noise} || n {n}')
        m = AnisotropicSet(weights=t.forwd.dimWeights(), cardinality=n, verbose=verbose, save=save)
        e = None
        for mode in ['cheby'] :
            s = Legendre(multis=m, target=t, verbose=verbose, dist=mode, save=save)
            e = s.computeError(accurc=.001, max_n=1e5, verbose=verbose)

        if e is not None and e.hedist < 1e-8 :
            break


if __name__ == "__main__" :

    n_targets = 5
    l = [2**i for i in range(7)]
    ds = [sum(l[:i]) for i in range(1,8)]
    alphas = [1, 2]
    noises = [.1, .5]
    wkerns = [10]
    xmeas = np.linspace(-.9, .9, 10)

    util.log.print_start('Setting up target densities\n')
    start = datetime.datetime.now()

    try :
        with ThreadPoolExecutor(max_workers=1) as executor :

            for (d, noise, alpha, wkern) in itertools.product(ds, noises, alphas, wkerns) :

                forwd = Convolution(basis=util.basis.hats, dim=d, alpha=alpha, wkern=wkern, xmeas=xmeas, save=True)
                posts = GaussianPosterior.fromConfig(fwd=forwd, noise=noise)
                n_prx = min(len(posts), n_targets)

                ts = posts[:n_prx]
                for _ in range(n_prx, n_targets) :
                    ts += [GaussianPosterior(forwd=copy.deepcopy(forwd), truep=util.random.points(d), noise=noise, save=True)]

                for t in ts :
                    if QUIT : break
                    #executor.submit(create_surrogates, t)
                    create_surrogates(t)

    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))
