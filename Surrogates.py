import numpy as np
import time

import MultiIndex as mi
import Database as db
import Densities as de

import legendreutil, randutil, pointutil, require


class Legendre :

    def __init__(self, *, multis, target, pmode='cheby', pmode_det=None, n='wls', verbose=0, save=False, domain=None) :
        require.equal(multis.dim, target.dim, 'multis.dim', 'target.dim')
        self.multis = multis
        self.target = target
        self.coeffs = None
        self.dim = multis.dim
        self.domain = domain

        if save :
            self.dbo, isnew = db.SurrogateDBO.get_or_create(
                target=target.name, target_id=None if not hasattr(target, 'dbo') else target.dbo.id, multis=multis.name, multis_id=multis.dbo.id, pmode=pmode)
            if not isnew and self.dbo.coeffs is not None :
                self.coeffs = db.fr_string(self.dbo.coeffs)

        if self.coeffs is None :
            if verbose > 0 : print('\t SETUP Surrogate')
            start = time.process_time()

            points, weights = pointutil.get_sample_points_and_weights(multis, pmode, pmode_det, n)
            lhs = legendreutil.evaluate_basis(points, multis, mode='old')
            if domain is not None :
                points = pointutil.scale(points.T, [[-1,1]]*self.dim, domain.r).T
            rhs = np.squeeze(target.evalSqrt(points))

            if weights is not None:
                for i in range(lhs.shape[0]) :
                    lhs[i,:] *= weights[i]
                    rhs[i]   *= weights[i]

            self.coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
            self.L = lhs
            self.G = np.dot(self.L.T, self.L)/self.L.shape[0] - np.eye(multis.size())
            if verbose > 1 : print('\t\t solved lstsq')

            #if lhs.size > 0 :
            #    if save : self.dbo.condnr = np.linalg.cond(lhs)
            if save :
                self.dbo.ctime = time.process_time() - start
                self.dbo.nevals = points.shape[1]
                self.dbo.coeffs = db.to_string(self.coeffs)
                self.dbo.save()
            if verbose > 0 : print('\t SETUP Surrogate DONE')

        require.equal(self.coeffs.shape, (multis.size(),), 'coeffs.shape', 'len(multis)')
        self.norm_lebesgue = self.coeffs.T.dot(self.coeffs) # =^ L2 norm squared of sqrt(f) and L1 norm of f wrt to the Lebesgue measure
        self.norm = 2**(-self.target.dim) * self.norm_lebesgue # --"-- wrt to the uniform measure
        self.m = len(self.coeffs)

    @classmethod
    def fromDbo(cls, dbo) :
        multis = None
        if dbo.multis == 'sparse' :
            multis = mi.SparseSet.fromId(dbo.multis_id)
        return Legendre(multis=multis, target=de.GaussianPosterior.fromId(dbo.target_id), method=dbo.method)

    def evalSqrt(self, x) :
        x = pointutil.ensure_shape(x, self.dim)
        if self.domain is not None :
            # assert x in self.domain
            x = pointutil.scale(x.T, self.domain.r, [[-1,1]]*self.dim).T
        basis = legendreutil.evaluate_basis(x, self.multis)
        return np.dot(basis, self.coeffs)

    def evalSqrtNrmd(self, x) :
        if self.norm == 0 : return 0
        return self.evalSqrt(x) / np.sqrt(self.norm)

    def eval(self, x) :
        return self.evalSqrt(x)**2

    def computeError(self, n=100, max_n=1e5, accurc=.01, verbose=False) :
        return SurrogateEvaluation(surrog=self, n=n, max_n=max_n, accurc=accurc, verbose=verbose, save=hasattr(self, 'dbo'))

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
                points = randutil.points(d, n)
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


if __name__ == '__main__' :
    import Densities as de
    import logutil

    logutil.print_start('Testing Surrogate Module...', end='\n')

    for save in [True, False] :

        for t in [de.Gaussian(mean=randutil.points(3,1), cova=randutil.covarm(3), save=save)] :  #de.generate_densities(save) :
            m = mi.TotalDegreeSet(dim=t.dim, order=int(np.ceil(50**(1/t.dim))), save=save)

            for mode in ['cheby'] :   #['ls', 'wls', 'ip_leja', 'ip_leggaus', 'wls_leja', 'wls_leggaus'] :
                s = Legendre(multis=m, target=t, pmode=mode, save=save)
                s.eval(randutil.points(t.dim))
                e = s.computeError()
                logutil.print_indent(t.name.ljust(15) + str(t.dim).ljust(3) + mode.ljust(12) + str(e.nevals).ljust(6) + ' {:.4f}'.format(e.hedist))
                e.deleteDbo()
                s.deleteDbo()
            t.deleteDbo()
            m.deleteDbo()

    logutil.print_done()
