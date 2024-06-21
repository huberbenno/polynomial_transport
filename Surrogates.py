import numpy as np
import time

import util
import MultiIndex as mi
import Database as db
import Densities as de


class Legendre :
    """ Surrogate $\\tilde f$ of a (unnormalized density) function $f : [-1,1]^d \\to [0,\\infty)$. """

    def __init__(self, *, target: de.TargetDensity, multis: mi.MultiIndexSet,
                 mode='wls', dist='cheby', verbose=0, save=False, domain=None) :
        """
        Parameters:
        -----------
        target : TargetDensity
        multis : MultiIndexSet
            The multi-index set $\\Lambda$ defining the polynomial ansatz space in which the approximation to
            $\\sqrt{f}$ is constructed.
        mode : str
            Approximation method determining the number of support points in terms of m := multis.size().
            Valid choices are
            - 'ip' : interpolation, uses m support points
            - 'wls' : weighted least squares, uses 10 * m * int(log(m)) support points
            - 'ls' : least squares, not recommended since it uses m**2 support points
        dist : str
            Distribution of support points. Valid choices are
            - 'leja' : Deterministic Leja points. Only works with mode='ip'.
            - 'leggaus' : Deterministic Legendre Gauss points. Only works with mode='ip'.
            - 'cheby' : Random points distributed according to the Chebychev measure.
            - 'cheby_ss' : As 'cheby', but the set of points (and thus the number of necessary density evaluations)
                           is reduced via the BSS algorithm (see https://arxiv.org/abs/2202.12625).
                           Requires installation of https://github.com/felixbartel/BSSsubsampling.jl.
                           Only works with mode='wls'.
            - 'christoffel' : Random points distributed according to the Christoffel measure.
                              Construction is computationally expensive.
            - 'uni' : Uniformly distributed points, not recommended.

        """
        if verbose > 0: print('\t SETUP Surrogate')
        util.require.equal(multis.dim, target.dim, 'multis.dim', 'target.dim')
        assert mode in ['ip', 'wls', 'ls']
        assert dist in ['leja', 'leggaus', 'cheby', 'cheby_ss', 'christoffel', 'uni']
        self.multis = multis
        self.target = target
        self.coeffs = None
        self.dim = multis.dim
        self.domain = domain

        if save :
            self.dbo, isnew = db.SurrogateDBO.get_or_create(
                target=target.name, target_id=None if not hasattr(target, 'dbo') else target.dbo.id,
                multis=multis.name, multis_id=multis.dbo.id, mode=mode, dist=dist)
            if not isnew and self.dbo.coeffs is not None :
                self.coeffs = db.fr_string(self.dbo.coeffs)

        if self.coeffs is None :
            start = time.process_time()

            # Determine number of support nodes
            n = multis.size()
            if mode == 'wls' : n = 10 * n * int(np.log(n))
            if mode == 'ls' :  n = n ** 2

            # Determine support points and weights
            points, weights = util.points.get_sample_points_and_weights(multis, dist, n)

            # Compute LHS and RHS
            lhs = util.legendre.evaluate_basis(points, multis, mode='old')
            if domain is not None : points = util.points.scale(points.T, [[-1, 1]] * self.dim, domain.r).T
            rhs = np.squeeze(target.evalSqrt(points))
            if weights is not None:
                for i in range(lhs.shape[0]) :
                    lhs[i,:] *= weights[i]
                    rhs[i]   *= weights[i]

            # Solve linear system
            self.coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
            ctime = time.process_time() - start

            self.L = lhs
            self.G = np.dot(self.L.T, self.L)/self.L.shape[0] - np.eye(multis.size())

            if save :
                self.dbo.condnr = 0 if lhs.size == 0 else np.linalg.cond(lhs)
                self.dbo.ctime = ctime
                self.dbo.nevals = points.shape[1]
                self.dbo.coeffs = db.to_string(self.coeffs)
                self.dbo.save()

        util.require.equal(self.coeffs.shape, (multis.size(),), 'coeffs.shape', 'len(multis)')

        # L2 norm squared of sqrt(f) and L1 norm of f wrt to the Lebesgue measure
        self.norm_lebesgue = self.coeffs.T.dot(self.coeffs)
        # L2 norm squared of sqrt(f) and L1 norm of f wrt to the uniform measure
        self.norm = 2**(-self.target.dim) * self.norm_lebesgue

        if verbose > 0: print('\t SETUP Surrogate DONE')

    def evalSqrt(self, x) :
        x = util.points.ensure_shape(x, self.dim)
        if self.domain is not None :
            # assert x in self.domain
            x = util.points.scale(x.T, self.domain.r, [[-1, 1]] * self.dim).T
        basis = util.legendre.evaluate_basis(x, self.multis)
        return np.dot(basis, self.coeffs)

    def evalSqrtNrmd(self, x) :
        if self.norm == 0 : return 0
        return self.evalSqrt(x) / np.sqrt(self.norm)

    def evalNrmd(self, x) :
        return self.eval(x) / self.norm_lebesgue

    def eval(self, x) :
        return self.evalSqrt(x)**2

    def computeError(self, n=100, max_n=1e5, accurc=.01, verbose=False) :
        return SurrogateEvaluation(surrog=self, n=n, max_n=max_n, accurc=accurc, verbose=verbose,
                                   save=hasattr(self, 'dbo'))

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()

    #TODO move legendreutil.evaluate_basis here


class SurrogateEvaluation :

    def __init__(self, *, surrog, n=100, max_n=1e5, accurc=.01, verbose=False, save=False) :
        if verbose > 0 : print('\t SETUP Surrogate Evaluation')
        self.surrog = surrog
        self.hedist = np.inf
        self.accurc = accurc

        if save :
            self.dbo, isnew = db.SurrogateEvalDBO.get_or_create(**{'surrog' : surrog.dbo.id})
            if not isnew :
                self.hedist = self.dbo.hedist
                self.nevals = self.dbo.nevals
                self.accurc = self.dbo.accurc

        if self.hedist is None or self.hedist == np.inf :
            d = surrog.target.dim
            surrog.target.computeNorm(accurc=accurc)
            hedist_samples = np.array([])
            approx_samples = np.array([])
            start = time.process_time()
            mean_tar = 0
            mean_sur = 0
            variance = np.inf
            while variance > accurc and len(approx_samples) < max_n :  # 2**32 :
                if verbose > 1 : print(variance, len(approx_samples))
                points = util.random.points(d, n)
                evals_tar = surrog.target.evalSqrt(points)
                evals_sur = surrog.evalSqrt(points)
                mean_tar += np.sum(evals_tar**2)
                mean_sur += np.sum(evals_sur**2)
                approx_samples_new = (evals_sur - evals_tar)**2
                hedist_samples_new = (evals_sur / np.sqrt(surrog.norm) - evals_tar / np.sqrt(surrog.target.norm))**2
                #print(hedist_samples.shape, hedist_samples_new.shape)
                approx_samples = np.concatenate((approx_samples, approx_samples_new.flatten()))
                hedist_samples = np.concatenate((hedist_samples, hedist_samples_new.flatten()))

                self.hedist = np.sqrt(np.mean(hedist_samples))
                self.approx = np.sqrt(np.mean(approx_samples))

                variance   = np.sqrt(np.mean((approx_samples - self.approx)**2) / (len(approx_samples) - 1))
                if verbose : print(len(approx_samples), self.approx, variance)
                n *= 2

            self.ctime = time.process_time() - start
            self.nevals = len(approx_samples)
            #mean_sur /= self.nevals
            #mean_tar /= self.nevals
            #print('Mean Target: ', mean_tar, surrog.target.norm_lebesgue, mean_tar/surrog.target.norm_lebesgue)
            #print('Mean Surrog: ', mean_sur, surrog.norm_lebesgue, mean_sur/surrog.norm_lebesgue)

            if save :
                self.dbo.approx = self.approx
                self.dbo.hedist = self.hedist
                self.dbo.nevals = self.nevals
                self.dbo.accurc = self.accurc
                self.dbo.ctime = self.ctime
                self.dbo.save()
        if verbose > 0 : print('\t SETUP Surrogate Evaluation DONE')

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()
