import time, copy
import numpy as np
from scipy import stats
from sobol import i4_sobol
import mc

def get_lattice(z, N, increment_only=False) :
    z = np.array(z)
    if increment_only :
        n = int(N/2)
        p = np.zeros((n,len(z)))
        for k in range(n) :
            v = 2.*(k+.5)/N * z
            p[k,:] = [vi % 1.0 for vi in v]
        return p
    else :
        p = np.zeros((N,len(z)))
        for k in range(N) :
            v = float(k)/N * z
            p[k,:] = [vi % 1.0 for vi in v]
        return p

def get_sobol(s, N, increment_only=False) :
    if increment_only :
        n = int(N/2)
        p = np.zeros((n,s))
        for j in range(n):
            p[j,:] = i4_sobol (s, j + n)[0]
        return p
    else :
        p = np.zeros((N,s))
        for k in range(N) :
            p[k,:] = i4_sobol(s, k)[0]
        return p

def shift(p) :
    random_vector = np.random.uniform(size=(p.shape[1],))
    for i in range(p.shape[0]) :
        p[i,:] += random_vector
        p[i,:] = [vi % 1.0 for vi in p[i,:]]
    return p

# Stolen from https://pythonhosted.org/UQToolbox/_modules/UQToolbox/sobol_lib.html
def scramble(X):
    """
    Scramble function as in Owen (1997)

    Reference:
    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """

    N = len(X) - (len(X) % 2)
    N_half = int(N/2)

    idx = X[0:N].argsort()
    iidx = idx.argsort()

    # Generate binomial values and switch position for the second half of the array
    bi = stats.binom(1,0.5).rvs(size=N_half).astype(bool)
    pos = stats.uniform.rvs(size=N_half).argsort()

    # Scramble the indexes
    tmp = idx[0:N_half][bi];
    idx[0:N_half][bi] = idx[N_half:N][pos[bi]];
    idx[N_half:N][pos[bi]] = tmp;

    # Apply the scrambling
    X[0:N] = X[0:N][idx[iidx]];

    # Apply scrambling to sub intervals
    if N > 2:
        X[0:N_half] = scramble(X[0:N_half])
        X[N_half:N] = scramble(X[N_half:N])

    return X

def qmc_points(*, mode, randomization, s, m, increment_only) :
    points = None
    if mode == 'sobol' :
        points = get_sobol(s, 2**m, increment_only)
    else :
        z = [1, 433461, 472323, 440637, 231645, 275007, 113895, 331051, 283181, 384579, 288619, 306439, 309943, 452525, 319841, 217509, 84615, 111067, 374949, 315005, 369473,  95709, 155273, 215539, 486377, 107399, 203705, 168683]
        points = get_lattice(z[:s], 2**m, increment_only)
    if randomization is None :
        return points
    else :
        return randomize_qmc_points(points=points, randomization=randomization)

def randomize_qmc_points(*, points, randomization) :
    points_copy = copy.deepcopy(points) # make a deep copy since randomization affects input point array
    if randomization == 'shift' :
        return shift(points_copy)
    if randomization == 'scramble' :
        return np.array([scramble(points_copy[:, i]) for i in range(points_copy.shape[1])]).transpose()

def qmc(*, func, s, n, mode='lattice', randomization='shift', increment_only=False) :
    start = time.process_time()
    points = qmc_points(mode=mode, randomization=randomization, s=s, m=int(np.ceil(np.log2(n))), increment_only=increment_only)
    samples = [func(points[i]) for i in range(len(points))]
    return np.mean(samples), time.process_time() - start

def qmc_randomized(*, func, s, m, mode='lattice', randomization='shift', r=10, point_offset=0) :
    start = time.process_time()
    estimates = np.zeros((r,))
    points = qmc_points(mode=mode, randomization=None, s=s, m=m, increment_only=False) + point_offset
    for i in range(r) :
        points_random = randomize_qmc_points(points=points, randomization=randomization)
        estimates[i] = np.mean([func(points_random[i]) for i in range(len(points_random))])
    return np.mean(estimates), mc.sample_var_estimate(estimates), time.process_time() - start
