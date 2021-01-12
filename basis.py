import numpy as np
import matplotlib.pyplot as plt

import plotutil

def hats(x, p, alpha=1) :
    d = len(p)
    interval = [-1, 1]
    midpoint = 0
    res = 0
    i = 1
    while i <= d :
        coeff = 2**(-alpha*(np.floor(np.log2(i)))) * p[i-1]
        if x < midpoint :
            res += coeff * (x - interval[0]) / (midpoint - interval[0])
            interval = [interval[0], midpoint]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i
        else :
            res += coeff * (interval[1] - x) / (interval[1] - midpoint)
            interval = [midpoint, interval[1]]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i + 1
    return res

def hats_cdec(x, p, alpha=1) :
    d = len(p)
    interval = [-1, 1]
    midpoint = 0
    res = 0
    i = 1
    while i <= d :
        coeff = 2**(-alpha*i) * p[i-1]
        if x < midpoint :
            res += coeff * (x - interval[0]) / (midpoint - interval[0])
            interval = [interval[0], midpoint]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i
        else :
            res += coeff * (interval[1] - x) / (interval[1] - midpoint)
            interval = [midpoint, interval[1]]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i + 1
    return res

def steps(x, p, alpha=1) :
    d = len(p)
    interval = [-1, 1]
    midpoint = 0
    res = 0
    i = 1
    while i <= d :
        coeff = 2**(-alpha*(np.floor(np.log2(i)))) * p[i-1]
        if x < midpoint :
            res += coeff
            interval = [interval[0], midpoint]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i
        else :
            res -= coeff
            interval = [midpoint, interval[1]]
            midpoint = (interval[1] + interval[0])/2
            i = 2 * i + 1
    return res

if __name__ == '__main__' :
    fig = plt.figure()
    x = np.linspace(-1,1,501)

    ax1 = plotutil.get_ax(fig, 2, 1, title='hats')
    ax1.plot(x, [hats_cdec(xi, [1]) for xi in x], label='l=0')
    for l in range(1,7) :
        ax1.plot(x, [hats_cdec(xi, [0] * (2**l - 1) + [1] * 2**l) for xi in x], label='l={}'.format(l))
    ax1.legend()

    ax2 = plotutil.get_ax(fig, 2, 2, title='steps')
    ax2.plot(x, [steps(xi, [1]) for xi in x], label='l=0')
    for l in range(1,7) :
        ax2.plot(x, [steps(xi, [0] * (2**l - 1) + [1] * 2**l) for xi in x], label='l={}'.format(l))
    ax2.legend()
    plt.show()
