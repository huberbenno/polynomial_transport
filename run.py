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

def create_data(a) :
    t, m = a
    s = Legendre(multis=m, target=t, method='wls')
    e = SurrogateEvalDBO.get_or_create_from_args(t, s, 'mc', save=True)

def create_args(n=10) :

    iters = []

    d2k = {1 : 40, 2 : 30 , 3 : 15}

    for d, k in d2k.items() :
        iters.append(itertools.product(
            [Gaussian(mean=randutil.points(d,1), cova=randutil.covarm(d)) for _ in range(n)],
            [TotalDegreeSet(dim=d, k=ki) for ki in range(2, k)]
        ))

    return itertools.chain(*iters)
#-------------------------------------------------------------------------------
#    Main
#-------------------------------------------------------------------------------

if __name__ == "__main__" :

    start = datetime.datetime.now()

    try :
        with ProcessPoolExecutor(max_workers=5) as executor :

            for t, m in create_args() :
                if QUIT : break
                #executor.submit(create_data, (t,m))
                create_data((t, m))

    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))


