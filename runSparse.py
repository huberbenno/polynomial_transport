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
    while n < 1000 :
        m = SparseSet.withSize(weights=t.weights, n=n, t=threshold, verbose=verbose)
        s = Legendre(multis=m, target=t, method='wls', verbose=verbose)
        e = SurrogateEvalDBO.get_or_create_from_args(t, s, 'mc', save=True, verbose=verbose)
        print(t.dim, lim, m.cardinality, e.hedist)
        if e.hedist < lim :
            break
        else :
            n += base
            if n == 10 * base :
                base *= 10
            threshold = m.threshold


def create_args(ds, n) :
    tars = []
    for d in ds :
        i = 0
        for row in DB.execute_sql('select mean, cova from gaussiandbo where dim = {}'.format(d)).fetchall() :
            mean = fr_string(row[0])
            cova = fr_string(row[1]).reshape((mean.shape[0], mean.shape[0]))
            #print(mean, cova)
            #if len(cova.shape) == 1 : cova = np.expand_dims(cova, axis=1)
            tars.append(DyingGaussian(mean=mean))
            i += 1
            if i >= n : break
        for j in range(i, n) :
            tars.append(DyingGaussian(mean=randutil.points(d,1)))
    return tars
#-------------------------------------------------------------------------------
#    Main
#-------------------------------------------------------------------------------

if __name__ == "__main__" :

    start = datetime.datetime.now()

    try :
        with ProcessPoolExecutor(max_workers=6) as executor :

            for lim in [1e-10] :

                for t in create_args([1,2,3,4,5,10][::-1], 1) :

                    if QUIT : break
                    executor.submit(create_data, (t, lim))
                    #create_data((t,lim))


    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))


