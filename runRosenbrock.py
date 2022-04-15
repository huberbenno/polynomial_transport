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

def create_args() :

    ks = range(40,2,-2)
    aa = [0, .1, .5, 1]
    bb = [1, 10, 100]

    return itertools.product(
        [Rosenbrock(a=a, b=b) for a in aa for b in bb],
        [TotalDegreeSet(dim=2, k=k) for k in ks]
    )

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


