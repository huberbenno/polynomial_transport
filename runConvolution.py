#!/usr/bin/python3
import sys, os, shutil, subprocess, glob, time, datetime, argparse, itertools
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from Database import *
from Densities import *
from MultiIndex import *
from Surrogates import *

import randutil

QUIT = False

def create_data(a, verbose=False) :
    t, lim = a
    n = 3
    base = 1
    threshold = 1
    oldcardinality = 0
    while n < 1000 :
        print(n, base, threshold)
        if QUIT : break
        m = SparseSet.withSize(weights=t.forwd.weights, n=n, t=threshold, verbose=False)
        if m.cardinality > oldcardinality :
            oldcardinality = m.cardinality
            s = Legendre(multis=m, target=t, method='wls', verbose=verbose)
            e = SurrogateEvalDBO.get_or_create_from_args(t, s, 'mc', save=True, verbose=verbose)
            print('dim: ', t.dim, ' - noise: ', t.forwd.dbo.noise, ' - alpha: ', t.forwd.alpha,
                  ' - size: ', m.cardinality, ' - l2dist: ', e.l2dist)
        if e.l2dist < lim :
            break
        else :
            n += base
            if n == 10 * base :
                base *= 10
            threshold = m.threshold


def create_args() :
    n = 3
    ds = [5,10,20,40,80]
    alphas = [3,5]
    sigmas = [.5,1]
    basiss = [basis.hats_cdec]
    xeval = np.linspace(-1,1,7)

    tars = []
    for (d, noise, alpha, base) in itertools.product(ds, sigmas, alphas, basiss) :
        print(d, noise, alpha, base)
        i = 0
        forwd = fw.Convolution(basis=base, dim=d, alpha=alpha, noise=noise)
        for row in DB.execute_sql('select truep, mean from gaussianposteriordbo pos join gaussiandbo gauss on pos.gauss_id = gauss.id join forwarddbo as fwd on pos.forwd_id = fwd.id'
                                  + ' where forwd_id = {} and xeval = \'{}\' and basis = \'{}\''.format(forwd.dbo.id, to_string(xeval), base.__name__)).fetchall() :
            print(d, noise, alpha, 'old')
            t = GaussianPosterior(forwd=forwd, truep=fr_string(row[0]), xeval=xeval, xmsrmt=fr_string(row[1]), noise=noise)
            tars.append(t)
            i += 1
            if i >= n : break
        for j in range(i, n) :
            print(d, noise, alpha, 'new')
            truep = randutil.points(d)
            xmsrmt = forwd.eval(truep, xeval=xeval) + noise*np.random.randn(len(xeval),1)
            t = GaussianPosterior(forwd=forwd, truep=truep, xeval=xeval, xmsrmt=xmsrmt, noise=noise)
            tars.append(t)
    return tars


if __name__ == "__main__" :

    start = datetime.datetime.now()

    try :
        with ProcessPoolExecutor(max_workers=6) as executor :

            for lim in [1e-16] :

                for t in create_args() :

                    if QUIT : break
                    executor.submit(create_data, (t, lim))
                    #create_data((t,lim))


    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))
