import numpy as np
import time

import util
import MultiIndex as mi
import Database as db
import Densities as de


class SurrogateEvaluation :

    def __init__(self, *, surrog, n=100, max_n=1e5, accurc=.01, verbose=False, save=False) :
        if verbose > 0 : print('\t Evaluation / ', end='')
        self.surrog = surrog
        self.hedist = np.inf
        self.accurc = accurc

        if save :
            self.dbo, isnew = db.SurrogateEvalDBO.get_or_create(**{'surrog' : surrog.dbo.id})
            if not isnew :
                if verbose > 0: print('loading / ', end='')
                self.approx = self.dbo.approx
                self.hedist = self.dbo.hedist
                self.nevals = self.dbo.nevals
                self.accurc = self.dbo.accurc

        if self.hedist is None or self.hedist == np.inf :
            if verbose > 0: print('computing / ', end='')

            hedist_samples = np.array([])
            approx_samples = np.array([])
            start = time.process_time()
            variance = np.inf

            while variance > accurc and len(approx_samples) < max_n :
                points = util.random.points(surrog.target.dim, n)
                evals_tar = surrog.target.evalSqrt(points)
                evals_sur = surrog.evalSqrt(points)

                approx_samples_new = (evals_sur - evals_tar)**2
                hedist_samples_new = (evals_sur / np.sqrt(surrog.norm) - evals_tar / np.sqrt(surrog.norm))**2
                approx_samples = np.concatenate((approx_samples, approx_samples_new.flatten()))
                hedist_samples = np.concatenate((hedist_samples, hedist_samples_new.flatten()))

                self.hedist = np.sqrt(np.mean(hedist_samples))
                self.approx = np.sqrt(np.mean(approx_samples))

                variance = np.sqrt(np.mean((hedist_samples - self.hedist)**2) / (len(hedist_samples) - 1))
                if verbose > 1 : print(f'\t\t {len(approx_samples)}, {self.approx:.2e}, {self.hedist:.2e}, {variance:.2e}')
                n *= 2

            self.ctime = time.process_time() - start
            self.nevals = len(approx_samples)
            if verbose > 0 : print(f'd_H={self.hedist:.2e}, var={variance:.2e}, n={self.nevals} /', end='')

            if save :
                self.dbo.approx = self.approx
                self.dbo.hedist = self.hedist
                self.dbo.nevals = self.nevals
                self.dbo.accurc = self.accurc
                self.dbo.ctime = self.ctime
                self.dbo.save()
        if verbose > 0 : print('done.')

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()
