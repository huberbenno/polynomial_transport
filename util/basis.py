import numpy as np
import matplotlib.pyplot as plt


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
        coeff = 2**(-alpha*i + 1) * p[i-1]
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

