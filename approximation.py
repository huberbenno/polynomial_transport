import numpy as np

import pymuqModeling_ as mm
import pymuqUtilities as mu
import pymuqApproximation as ma

import lejautil, legendreutil

def pce(target, d, multiset, method) :
    if method == 'smolyak' :
        smolyPCE = ma.AdaptiveSmolyakPCE(target, [ma.LejaQuadrature()]*d, [ma.Legendre()]*d, 'interpolation')
        pce = smolyPCE.Compute(multiset, {'shouldAdapt' : 0})
        coeffs = pce.GetCoeffs()[0]
        assert(multiset.Size() == len(coeffs))
        return coeffs

    samples = None
    weights = None
    m = multiset.Size()
    if method == 'ls' :
        samples = np.random.uniform(low=-1, high=1, size=(10*m**2,args.d))
    elif method == 'wls' :
        samples = np.sin(np.random.uniform(low=-3*np.pi/2, high=np.pi/2,
                                           size=(10*m*int(np.log(m)),d)))
        weights = np.pi/2*np.array([np.prod([np.sqrt(1-x**2) for x in samples[i,:]]) for i in range(samples.shape[0])])
    elif method == 'ip' :
        samples = lejautil.leja_points(multiset, d)
    else : assert(False)
    rhs = np.array([target.Evaluate([p])[0][0] for p in samples])
    lhs = legendreutil.evaluate_basis(samples.T, multiset)
    if weights is not None:
        for i in range(lhs.shape[0]) :
            lhs[i,:] *= weights[i]
        rhs = np.multiply(rhs, weights)

    coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coeffs
