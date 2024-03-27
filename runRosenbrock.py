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

def create_data(a, save=True) :
    t, m = a
    s = Legendre(multis=m, target=t, save=save)
    e = s.computeError(accurc=.0001)

def create_args(save=True) :

    ks = range(10,2,-2)
    delta = 10*np.pi/72
    thetas = [2*np.pi+.3-i*2/3*np.pi-delta for i in range(3)]
    cs = [.3*np.array([np.cos(th), np.sin(th)-.2]) for th in [(6 + i*8)/12*np.pi+delta for i in range(3)]]

    t_m = MultimodalDensity(
            densities=[Rosenbrock(a=.4, b=4, theta=t, centr=c, scale=3.5, save=save) for c, t in zip(cs,thetas)],
            weights=[1,1,1])

    return itertools.product(
        [t_m],
        [TotalDegreeSet(dim=2, order=k, save=save) for k in ks]
    )


if __name__ == "__main__" :

    save = False

    start = datetime.datetime.now()

    try :
        with ProcessPoolExecutor(max_workers=5) as executor :

            for t, m in create_args(save) :
                if QUIT : break
                #executor.submit(create_data, (t,m))
                create_data((t, m), save)

    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))


