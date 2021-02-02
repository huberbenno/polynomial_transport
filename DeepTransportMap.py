import time, copy
import numpy as np
import matplotlib.pyplot as plt

import pymuqModeling_ as mm
import pymuqUtilities as mu
import pymuqApproximation as ma

import legendreutil, testutil, plotutil, approximation

from GaussianPosterior import GaussianPosterior
from TransportMap import TransportMap

def S_bar_k_inv(maps, x) :
    for m in maps[::-1] : x = m.inverse_evaluate(x)
    return x

def S_bar_k_list(maps, x) :
    if not maps : return x
    res = [maps[0].evaluate(x)]
    for m in maps[1:] : res.append(m.evaluate(res[-1]))
    return res

def g_bar_x(maps, x) :
    S_bar_k_list_x = S_bar_k_list(maps[:-1], x)
    g_bar_terms = [maps[i].det_dS(S_bar_k_list_x[i-1]) for i in range(1,len(maps))]
    return maps[0].det_dS(x) * np.prod(g_bar_terms)

class Target_k(mm.PyModPiece):

    def __init__(self, *, target, temperatures, maps):
        super(Target_k, self).__init__(target.inputSizes, target.outputSizes)
        self.target = target
        self.temperatures = temperatures
        self.maps = maps
        self.k = len(maps)

    def EvaluateImpl(self, inputs) :
        if self.k == 0 :
            self.outputs = [self.target.Evaluate(inputs)[0]**(self.temperatures[0]/2) ]
        else :
            S_bar_k_inv_x = S_bar_k_inv(self.maps, inputs[0])
            denominator = self.target.Evaluate([np.array(S_bar_k_inv_x)])[0]**self.temperatures[self.k]
            nominator = g_bar_x(self.maps, S_bar_k_inv_x)
            self.outputs = [ np.sqrt(denominator / nominator / 2) ]

class Target_k_simple(mm.PyModPiece):

    def __init__(self, *, target, temperatures, maps):
        super(Target_k_simple, self).__init__(target.inputSizes, target.outputSizes)
        self.target = target
        self.temperatures = temperatures
        self.maps = maps
        self.k = len(maps)

    def EvaluateImpl(self, inputs) :
        if self.k == 0 :
            self.outputs = [ self.target.Evaluate(inputs)[0]**(self.temperatures[0]/2) / np.sqrt(2) ]
        else :
            S_bar_k_inv_x = S_bar_k_inv(self.maps, inputs[0])
            self.outputs = [ self.target.Evaluate([np.array(S_bar_k_inv_x)])[0]**(self.temperatures[self.k-1]/2) / np.sqrt(2) ]

class DeepTransportMap :

    def __init__(self, d, target, temperatures, multiset, method='wls') :
        self.d = d
        self.multiset = multiset
        self.maps = []

        for t in temperatures :
            target_k = Target_k(target=target, temperatures=temperatures, maps=[m for m in self.maps])
            coeffs = approximation.pce(target_k, d, multiset, method)
            self.maps.append(TransportMap(self.d, coeffs, self.multiset))

    def evaluate(self, x) :
        return S_bar_k_inv(self.maps, x)

    def density_sqrt(self, x) :
        return np.sqrt(self.density(x))

    def density(self, x) :
        return g_bar_x(self.maps, x)


if __name__ == '__main__' :
    fig = plt.figure()
    nbins = 51
    nsamples = 1000
    x = np.linspace(-1, 1, nbins)

    temp_list = [[2**(-L+n) for n in range(L+1)] for L in range(4)]
    print(temp_list)
    l2_errs = []
    t_setup = []
    t_eval  = []

    d = 1

    if d == 1 :
        ax1 = plotutil.get_ax(fig, 3, 1, title='Densities')

        target   = GaussianPosterior(noise=.1, y_measurement=[0.])
        multiset = mu.MultiIndexFactory.CreateAnisotropic([.5], .05)
        print('Number of multi-indices : ', multiset.Size())

        true_target = np.array([target.Evaluate([np.array([xi])])[0][0] for xi in x])
        true_target /= np.sum(true_target)
        ax1.plot(x, true_target, 'k', linewidth = 4, label='target')
        for temps in temp_list :
            print(temps)
            start = time.process_time()
            deeptm = DeepTransportMap(1, target, temps, multiset)
            t_setup.append(time.process_time() - start)

            start = time.process_time()
            samples = [deeptm.evaluate([np.random.uniform(low=-1, high=1)])[0] for _ in range(nsamples)]
            t_eval.append((time.process_time() - start)/nsamples*10)

            tm_hist, _  =  np.histogram(samples, bins=nbins, density=True)
            tm_hist /= np.sum(tm_hist)
            ax1.plot(x, tm_hist, label=str(len(temps)))

            l2_errs.append(np.sqrt(np.sum((true_target - tm_hist)**2)))
        ax1.legend()

    elif d == 2 :
        ax1 = plotutil.get_ax(fig, 3, 1, title='Densities', projection='3d')

        target   = GaussianPosterior(noise=.1, y_measurement=[0., 0.])
        multiset = mu.MultiIndexFactory.CreateAnisotropic([.3, .3], .01)
        print('Number of multi-indices : ', multiset.Size())

        X, Y   = np.meshgrid(x, x)
        points = np.vstack((X.flatten(), Y.flatten())).transpose()

        true_target = np.array([target.Evaluate([xi])[0][0] for xi in points])
        true_target /= np.sum(true_target)

        levels = np.linspace(0, 0.1, 100)  #(z_min,z_max,number of contour),

        ax1.plot_surface(X, Y, true_target.reshape((len(x), len(x))), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))


        for temps in temp_list :
            print(temps)
            start = time.process_time()
            deeptm = DeepTransportMap(2, target, temps, multiset)
            t_setup.append(time.process_time() - start)

            start = time.process_time()
            samples = np.array([deeptm.evaluate(np.random.uniform(low=-1, high=1, size=(2,))) for _ in range(10)])
            t_eval.append(time.process_time() - start)

            tm_hist, _, _  =  np.histogram2d(samples[:,0], samples[:,1], bins=nbins, density=True)
            tm_hist /= np.sum(tm_hist)

            offset = 10*len(temps)
            ax1.plot_surface(X, Y, offset+tm_hist.reshape((len(x), len(x))), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

            l2_errs.append(np.sqrt(np.sum((true_target - tm_hist.flatten())**2)))

    depths = [len(temps) for temps in temp_list]
    ax2 = plotutil.get_ax(fig, 3, 2, title='L2 Error', xlabel='DIRT depth')
    ax2.set_xticks(depths)
    ax2.plot(depths, l2_errs)

    ax3 = plotutil.get_ax(fig, 3, 3, title='Runtime', xlabel='DIRT depth')
    ax3.set_xticks(depths)
    ax3.plot(depths, t_setup, label='setup time')
    ax3.plot(depths, t_eval, label='eval time per 10 samples')

    ax3.legend()
    plt.show()

