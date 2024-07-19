import numpy as np
import time

import util
import MultiIndex as mi
import Database as db
import Densities as de


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
