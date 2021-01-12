#!/usr/bin/python
import time, base64
import numpy as np
import peewee as pw

import pymuqUtilities as mu
import pymuqApproximation as ma

from Convolution import Convolution
from basis import hats, hats_cdec, steps
import legendreutil, approximation

#TODO unique constraints
DB = pw.SqliteDatabase('data.db')

def print_multiset(multiset) :
    print('Multiset ({} indices): '.format(len(multiset.GetAllMultiIndices())))
    for m in multiset.GetAllMultiIndices() : print(m.GetVector(), end=' ')
    print()
    print('Active Multiset ({} indices): '.format(multiset.Size()))
    for i in range(multiset.Size()) : print(multiset.IndexToMulti(i).GetVector(), end=' ')
    print()

def to_string(arr) : return base64.binascii.b2a_base64(arr).decode("ascii")
def fr_string(stg) : return np.frombuffer(base64.binascii.a2b_base64(stg.encode("ascii")))

class BaseModel(pw.Model) :
    class Meta:
        database = DB

class Model(BaseModel) :
    mtype = pw.TextField()
    dim   = pw.IntegerField()
    basis = pw.TextField()
    alpha = pw.DoubleField()
    noise = pw.DoubleField()
    x_val = pw.TextField()

    basis_map = {'hats'      : hats,
                 'hats_cdec' : hats_cdec,
                 'steps'     : steps}

    @staticmethod
    def get_or_create_from_args(args) :
        return Model.get_or_create(mtype = args.model,
                                   dim   = args.d,
                                   basis = args.basis,
                                   alpha = args.alpha,
                                   noise = args.sigma,
                                   x_val = to_string(np.linspace(-1,1,10)))[0]

    def get_model(self) :
        if self.mtype == 'convolution' :
            return Convolution(basis = self.basis_map[self.basis],
                               d     = self.dim,
                               alpha = self.alpha,
                               x_val = fr_string(self.x_val))

class Parameter(BaseModel) :
    model = pw.ForeignKeyField(Model, backref='param')
    p_id  = pw.IntegerField()
    value = pw.TextField()

   # class Meta:
   #     indexes = ((('model', 'p_id'), True))

    @staticmethod
    def get_or_create_from_args(model, args) :
        obj_args = {'model' : model.id, 'p_id' : args.p}
        obj = Parameter.get_or_none(**obj_args)
        if obj is None :
            value = np.random.uniform(low=-1, high=1, size=(args.d,))
            obj = Parameter.create(**obj_args, value=to_string(value))
            obj.save()
        return obj

class Measurement(BaseModel) :
    param = pw.ForeignKeyField(Parameter, backref='measurement')
    m_id  = pw.IntegerField()
    y_val = pw.TextField()

#    class Meta:
#        indexes = ((('param', 'm_id'), True))
#        primary_key = pw.CompositeKey('param', 'm_id')

    @staticmethod
    def get_or_create_from_args(model, model_operator, param, args) :
        obj_args = {'param' : param.id, 'm_id' : args.m}
        obj = Measurement.get_or_none(**obj_args)
        if obj is None :
            y_val = (  model_operator.Evaluate([fr_string(param.value)])
                     + model.noise*np.random.randn(len(model_operator.x_val)))
            obj = Measurement.create(**obj_args, y_val=to_string(y_val))
            obj.save()
        return obj

class Multiset(BaseModel) :
    stype = pw.TextField()
    dim   = pw.IntegerField()

    order = pw.IntegerField(null=True)
    alpha = pw.DoubleField(null=True)
    epsln = pw.DoubleField(null=True)

    size  = pw.IntegerField(null=True)
    ctime = pw.DoubleField(null=True)

    multiset = None

    @staticmethod
    def get_or_create_from_args(args) :
        multiset = None
        obj_args = {'stype' : args.stype, 'dim' : args.d}

        start = time.process_time()
        if args.stype == 'totalorder' :
            obj_args.update({'order': args.order})
            multiset = mu.MultiIndexFactory.CreateTotalOrder(args.d,args.order)
        elif args.stype == 'anisotropic' :
            obj_args.update({'alpha': args.alpha, 'epsln' : args.eps})
            weights = [2**(-args.alpha*np.ceil(np.log2(i+1))) for i in range(1,args.d+1)]
            multiset = mu.MultiIndexFactory.CreateAnisotropic(weights,args.eps)
        else : assert(False)
        obj_args.update({'ctime': time.process_time() - start,
                         'size' : multiset.Size()})

       # print('\nMultiset:')
       # print_multiset(multiset)

        obj, _ = Multiset.get_or_create(**obj_args)
        obj.multiset = multiset

        return obj

class PceCalculation(BaseModel) :
    # Parameters
    msrmt = pw.ForeignKeyField(Measurement, backref='calcs')
    multi = pw.ForeignKeyField(Multiset, backref='calcs')
    methd = pw.TextField()

    # Data
    coeff = pw.TextField(null=True)
    ctime = pw.DoubleField(null=True)

    coeff_array = None

    @staticmethod
    def get_or_create_from_args(msrmt, multi, target, args) :
        obj_params =  {'msrmt' : msrmt.id, 'multi' : multi.id, 'methd' : args.appr}
        obj, is_new = PceCalculation.get_or_create(**obj_params)
        if not is_new :
            obj.coeff_array = fr_string(obj.coeff)
            return obj

        start = time.process_time()
        obj.coeff_array = approximation.pce(target, args.d, multi.multiset, args.methd)
       # print('\nCoeffs:')
       # print(obj.coeff_array)
        obj.coeff = to_string(obj.coeff_array)
        obj.ctime = time.process_time() - start
        obj.save()
        return obj


class PceEvaluation(BaseModel):
    calc  = pw.ForeignKeyField(PceCalculation, backref='evals')
    nsamp = pw.IntegerField()
    methd = pw.TextField()

    # Data
    l2dst = pw.DoubleField(null=True)
    ctime = pw.DoubleField(null=True)

    @staticmethod
    def get_or_create_from_args(pcecalc, multi, target, args) :
        obj_params =  {'calc' : pcecalc.id, 'nsamp' : 1000, 'methd' : args.int}
        obj, is_new = PceEvaluation.get_or_create(**obj_params)
        if not is_new :
            return obj

        start = time.process_time()
        p_grid = None
        if args.int == 'grid' :
            grid = np.mgrid[tuple(slice(-1, 1.1, .1) for _ in range(args.d))]
            p_grid = np.vstack([grid[i].flatten() for i in range(args.d)])
        elif args.int == 'mc' :
            p_grid = np.random.uniform(low=-1, high=1, size=(args.d,1000))
        #elif a.int == 'qmc' :
        #    p_grid =
        else : assert(False)

        evals = np.array([target.Evaluate([p_grid[:,i]])[0][0] for i in range(p_grid.shape[1])])
        basis = legendreutil.evaluate_basis(p_grid, multi.multiset)
        predc = np.dot(basis, pcecalc.coeff_array)
        diff = np.subtract(evals, predc)

        obj.l2dst = np.sqrt(np.dot(diff.T, diff)/ p_grid.shape[1]) / np.sqrt( np.dot(evals, evals)/ p_grid.shape[1])
        obj.ctime = time.process_time() - start
        obj.save()

        return obj


if __name__ == '__main__' :
    DB.connect()
    DB.create_tables([Model, Parameter, Measurement, Multiset, PceCalculation, PceEvaluation]) #

    #model_params = {'mtype' : 'convolution', 'dim' : 5, 'basis' : 'hats', 'alpha' : 1, 'noise' : .1, 'x_val' : 'asdsa'}

    #model = Model.get_or_create(**model_params)[0]
    #param = Parameter.get_or_create(**{'model' : model.id, 'p_id' : 1, 'value' : '1 2 3'})[0]
    #msrmt = Measurement.get_or_create(param=param.id, m_id=1, y_val='1 2 3 5')[0]


    #test_calc = PceCalcData.create(model=model, basis='hats', alpha=1, dim=4, param='absxs', noise=.1, msrmt='adss',
      #                             epsln=.01, coeffs='adsadsad', mssize=50000, runtime=23423.2)
    #test_calc.save()
