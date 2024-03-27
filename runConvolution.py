#!/usr/bin/python3
import sys, os, shutil, subprocess, glob, time, datetime, argparse, itertools
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from Database import *
from Densities import *
from MultiIndex import *
from Surrogates import *
from Forward import *

import randutil, basis

QUIT = False

def create_surrogates(a, verbose=1, save=True) :
    t, lim = a
    n = 3
    base = 1
    threshold = 1
    oldcardinality = 2
    while n < 1000 :
        if QUIT : break
        m = SparseSet.withSize(weights=t.forwd.dimWeights(), n=n, t=threshold, verbose=0, save=save)
        e = None
        if m.cardinality > oldcardinality :
            oldcardinality = m.cardinality
            for mode in ['cheby'] :
                s = Legendre(multis=m, target=t, verbose=verbose, pmode=mode, save=save)
                e = s.computeError(accurc=.0001, verbose=verbose)
                print('dim: ', t.dim, ' - noise: ', t.noise, ' - alpha: ', t.forwd.alpha, ' - size: ', m.cardinality, ' - hedist: ', e.hedist)
        if e is not None and e.hedist < lim :
            break
        else :
            n += base
            if n == 10 * base :
                base *= 10
            threshold = m.threshold


def create_targets(save=True) :
    n_targets = 5
    l = [2**i for i in range(7)]
    ds = [sum(l[:i]) for i in range(1,8)][::-1]
    alphas = [2]
    noises = [.1]
    wkerns = [10]
    xmeas = np.linspace(-.9,.9,10)

    targets = []
    for (d, noise, alpha, wkern) in itertools.product(ds, noises, alphas, wkerns) :
        print(d, noise, alpha, wkern)
        forwd = Convolution(basis=basis.hats, dim=d, alpha=alpha, wkern=wkern, xmeas=xmeas, save=save)
        posts = GaussianPosterior.fromConfig(fwd=forwd, noise=noise)
        n_preex_targets = min(len(posts), n_targets)
        targets += posts[:n_preex_targets]
        targets += [GaussianPosterior(forwd=copy.deepcopy(forwd), truep=randutil.points(d), noise=noise, save=save) for _ in range(n_preex_targets, n_targets)]
    return targets

if __name__ == "__main__" :

    start = datetime.datetime.now()

    try :
        with ThreadPoolExecutor(max_workers=2) as executor :

            for lim in [1e-8] :

                for t in create_targets() :

                    if QUIT : break
                    #executor.submit(create_surrogates, (t, lim))
                    create_surrogates((t,lim))
                    #pass


    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))
