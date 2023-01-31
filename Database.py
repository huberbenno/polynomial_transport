#!/usr/bin/python
import time, base64
import numpy as np
import peewee as pw

import MultiIndex as mi
import Densities as ds
import Surrogates as sg

from scipy.stats import qmc

#TODO unique constraints
DB = pw.SqliteDatabase('data.db')
# DB = pw.SqliteDatabase(':memory:')

def to_string(arr) : return base64.binascii.b2a_base64(np.array(arr)).decode("ascii")
def fr_string(stg) : return np.frombuffer(base64.binascii.a2b_base64(stg.encode("ascii")))


class BaseModel(pw.Model) :
    class Meta:
        database = DB


# ---------- Forward Models --------------------

class ForwardDBO(BaseModel) :
    dim   = pw.IntegerField()
    basis = pw.TextField()
    alpha = pw.IntegerField()
    noise = pw.DoubleField()
    nquad = pw.IntegerField()
    # TODO kernel width, extra table for convolution

# ---------- Densities --------------------

class GaussianDBO(BaseModel) :
    dim  = pw.IntegerField()
    mean = pw.TextField()
    cova = pw.TextField()
    diag = pw.BooleanField()

class GaussianMmDBO(BaseModel) :
    dim  = pw.IntegerField()
    gauss1 = pw.ForeignKeyField(GaussianDBO)
    gauss2 = pw.ForeignKeyField(GaussianDBO)
    gauss3 = pw.ForeignKeyField(GaussianDBO, null=True)
    gauss4 = pw.ForeignKeyField(GaussianDBO, null=True)
    gauss5 = pw.ForeignKeyField(GaussianDBO, null=True)

class RosenbrockDBO(BaseModel) :
    a = pw.DoubleField()
    b = pw.DoubleField()
    theta = pw.DoubleField()
    centr = pw.TextField()
    scale = pw.DoubleField()

class GaussianPosteriorDBO(BaseModel) :
    forwd = pw.ForeignKeyField(ForwardDBO)
    gauss = pw.ForeignKeyField(GaussianDBO)
    truep = pw.TextField()
    xeval = pw.TextField()


# ---------- Indexsets --------------------

class MultiIndexSetDBO(BaseModel) :
    dim   = pw.IntegerField()
    mode  = pw.TextField()
    order = pw.IntegerField()

    size  = pw.IntegerField(null=True)
    ctime = pw.DoubleField(null=True)
    #TODO create or dump binaries?

class MultiIndexSetAnisotropicDBO(BaseModel) :
    dim    = pw.IntegerField()
    weight = pw.TextField()
    thresh = pw.DoubleField()

    size   = pw.IntegerField(null=True)
    ctime  = pw.DoubleField(null=True)


# ---------- Surrogate --------------------

class SurrogateDBO(BaseModel) :
    target = pw.TextField()
    target_id = pw.IntegerField()
    multis = pw.TextField()
    multis_id = pw.IntegerField()
    method = pw.TextField()
    condnr = pw.DoubleField(null=True)
    closei = pw.DoubleField(null=True)

    nevals = pw.IntegerField(null=True)
    coeffs = pw.TextField(null=True)
    ctime  = pw.DoubleField(null=True)

class SurrogateEvalDBO(BaseModel):
    surrog = pw.ForeignKeyField(SurrogateDBO, backref='evals')
    method = pw.TextField()

    # Data
    l2dist = pw.DoubleField(null=True)
    hedist = pw.DoubleField(null=True)
    nevals = pw.IntegerField(null=True)
    accurc = pw.DoubleField(null=True)
    ctime = pw.DoubleField(null=True)

    @staticmethod
    def get_or_create_from_args(target, surrog, method, nevals=1000, accurc=.001, save=False, verbose=True) :
        obj, is_new = SurrogateEvalDBO.get_or_create(**{'surrog' : surrog.dbo.id, 'method' : method})
        if not is_new and obj.l2dist is not None :
            return obj

        if verbose : print('Evaluation....', end=' ')
        obj.nevals = 0
        obj.l2dist = np.inf
        obj.hedist = np.inf
        variance = np.inf
        X = np.array([])
        Y = np.array([])
        norm = 1 #surrog.norm if surrog.norm > 0 else 1
        print('norm ', norm)

        start = time.process_time()

        while variance > accurc and obj.nevals < 50000 : #2**32 :
            points = None
            if obj.method == 'mc' :
                points = np.random.uniform(low=-1, high=1, size=(target.dim, nevals))
            elif obj.method == 'sobol' :
                points = 2*qmc.Sobol(target.dim, scramble=False).random(nevals).T - 1
            else : assert(False)
            obj.nevals += nevals

            evalsTar = target.eval(points)
            evalsSur = surrog.eval(points)

            X = np.concatenate((X, ((evalsTar - evalsSur)/norm)**2))
            Y = np.concatenate((Y, (np.sqrt(evalsTar) - np.sqrt(evalsSur))**2/norm)) #TODO check

            obj.l2dist = np.sqrt(np.mean(X))
            obj.hedist = np.sqrt(np.mean(Y))

            variance   = np.sqrt(np.mean((X - obj.l2dist)**2) / (obj.nevals - 1))
            print(obj.nevals, obj.l2dist, variance)

        obj.ctime = time.process_time() - start
        obj.accurc = accurc

        if save :
            obj.save()

        if verbose : print('Done')
        return obj


if __name__ == '__main__' :
    DB.connect()
    DB.create_tables([ForwardDBO, GaussianDBO, GaussianMmDBO, GaussianPosteriorDBO, RosenbrockDBO, MultiIndexSetDBO,
                      MultiIndexSetAnisotropicDBO, SurrogateDBO, SurrogateEvalDBO])

    args = {'dim' : 1, 'mean' : .1, 'cova' : .1, 'mode' : 'sum', 'order' : 5, 'methodAppr' : 'wls', 'methodEval' : 'mc'}
    #args = {'dim' : 2, 'mean' : [.1, .4], 'cova' : [[.5, 0], [0, .2]], 'mode' : 'sum', 'order' : 5, 'methodAppr' : 'wls'}
