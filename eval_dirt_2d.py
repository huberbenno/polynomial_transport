import sys, time, itertools
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

import pymuqUtilities as mu

import plotutil

from GaussianPosterior import GaussianPosterior
from DeepTransportMap import *

if __name__ == '__main__' :
    multis = [mu.MultiIndexFactory.CreateAnisotropic([.42, .41], 1/(10**i)) for i in range(1,15,3)]
    x_multis = [m.Size() for m in multis]
    n_samples = 3
    points = np.random.uniform(low=-1, high=1, size=(500,2))

    noise=.02
    target = GaussianPosterior(noise=noise , y_measurement=[.4,.4])
    true_target = np.array([target.Evaluate([xi])[0] for xi in points])[:,0]
    true_target /= np.sum(true_target)
    norm = np.sqrt(np.dot(true_target, true_target))

    temps = [[2**(-L+n) for n in range(L+1)] for L in range(2)]

    print('temps: ', temps)
    print('x_multis:', x_multis)

    results_l2 = np.zeros((len(temps), len(multis), n_samples))
    results_st = np.zeros((len(temps), len(multis), n_samples))
    results_et = np.zeros((len(temps), len(multis), n_samples))

    def compute(indices) :
        (i,j,k) = indices
        start = time.process_time()
        deeptm = DeepTransportMap(2, target, temps[i], multis[j], 'wls')
        st = time.process_time() - start

        start = time.process_time()
        density = [deeptm.density(xi) for xi in points]
        density /= np.sum(density)
        l2 = np.sqrt(np.sum((true_target - density)**2))/norm
        et = time.process_time() - start

        print('[', len(temps[i]), ',', multis[j].Size(), ',', st, ',', et, ']')
        return i, j, k, l2, st, et

    with ProcessPoolExecutor(max_workers=2) as executor:
        iterable = itertools.product(range(len(temps))[::-1], range(len(multis))[::-1], range(n_samples)[::-1])
        results = list(executor.map(compute, iterable))
        for (i, j, k, l2, st, et) in results :
            results_l2[i,j,k] = l2
            results_st[i,j,k] = st
            results_et[i,j,k] = et


    # Plotting
    fig = plt.figure()
    fig.suptitle('noise = {}'.format(noise), fontsize=20)

    ax = plotutil.get_ax(fig, 3, 1, title='L2 error', xlabel=r'$|\Lambda|$', logaxis=['x', 'y'])
    ax.set_xticks(x_multis)
    for i in range(len(temps)) :
        ax.errorbar(x_multis, np.mean(results_l2[i], axis=1), yerr=np.std(results_l2[i], axis=1), capsize=5, label='L = '+str(i))
    ax.legend()

    ax = plotutil.get_ax(fig, 3, 2, title='setup time', xlabel=r'$|\Lambda|$', logaxis=['x', 'y'])
    ax.set_xticks(x_multis)
    for i in range(len(temps)) :
        ax.errorbar(x_multis, np.mean(results_st[i], axis=1), yerr=np.std(results_st[i], axis=1), capsize=5, label='L = '+str(i))
    ax.legend()

    ax = plotutil.get_ax(fig, 3, 3, title='eval time', xlabel=r'$|\Lambda|$', logaxis=['x', 'y'])
    ax.set_xticks(x_multis)
    for i in range(len(temps)) :
        ax.errorbar(x_multis, np.mean(results_et[i], axis=1), yerr=np.std(results_et[i], axis=1), capsize=5, label='L = '+str(i))
    ax.legend()

    plt.tight_layout()
    plt.show()

    print('\n\nL2:')
    print(results_l2)
    print('\n\nSetup:')
    print(results_st)
    print('\n\nEval')
    print(results_et)
    print()
    plt.savefig('dirt_l2_noise={}.pdf'.format(noise), format='pdf')
