#!/usr/bin/python3
import sys, os, shutil, subprocess, glob, time, datetime, argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss

import pymuqUtilities as mu

from GaussianPosterior import *

from database import *
import basis, legendreutil, lejautil

QUIT = False
QUIT_ISO = False
QUIT_ANI = False

class Args() :
    def __init__(self) :
        self.model = 'convolution'
        self.d = 2
        self.p = 1
        self.m = 1
        self.basis = 'hats'
        self.appr = 'smolyak'
        self.int = 'mc'
        self.sigma = .1
        self.alpha = 1
        self.stype = 'totalorder'
        self.order = 0
        self.eps = .1
        self.cpu_count = 4

class SimpleTarget(mm.PyModPiece):

    def __init__(self, d):
        super(SimpleTarget, self).__init__([d], [1])

    def EvaluateImpl(self, inputs):
        self.outputs = [[inputs[0][0]*inputs[0][0]]]

def create_args() :

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='convolution', choices={'convolution'},
                        help='Model')
    parser.add_argument('-d', type=int, default=2,
                        help='Number of dimensions')
    parser.add_argument('-p', type=int, default=1,
                        help='Parameter Id')
    parser.add_argument('-m', type=int, default=1,
                        help='Measurement Id')
    parser.add_argument('-basis', type=str, default='hats', choices={'hats', 'hats_cdec', 'steps'},
                        help='Basis function type')
    parser.add_argument('-appr', type=str, default='wls', choices={'ls', 'wls', 'ip', 'smolyak'},
                        help='Approximation method')
    parser.add_argument('-int', type=str, default='mc', choices={'grid', 'mc', 'qmc'},
                        help='Integration method')
    parser.add_argument('-sigma', type=float, default=.1,
                        help='Noise variance')
    parser.add_argument('-alpha', type=int, default=1,
                        help='Basis function decay')
    parser.add_argument('-stype', type=str, default='totalorder', choices={'totalorder', 'anisotropic'},
                        help='Type of the multiindex set')
    parser.add_argument('-order', type=int, default=2,
                        help='Order of total order multi index set')
    parser.add_argument('-eps', type=float, default=.1,
                        help='Limit for dimension adaptive multi index set')
    parser.add_argument('-cpu-count', type=int, default=4,
                        help='Number of CPUs to use')
    return parser.parse_args()

def create_data(a) :

    global QUIT, QUIT_ISO, QUIT_ANI

    INFO = ''
    if   a.stype == 'totalorder'  : INFO = '[INFO_ISO_d={}_a={}_o={}_{}] '.format(a.d, a.alpha, a.order, a.appr)
    elif a.stype == 'anisotropic' : INFO = '[INFO_ANI_d={}_a={}_e={}_{}] '.format(a.d, a.alpha, a.eps, a.appr)

    if QUIT or (a.order and QUIT_ISO) or (a.eps and QUIT_ANI):
        print(INFO + ' Aborted.')
        return

    evals = (PceEvaluation.select().join(PceCalculation).join(Multiset)
                                   .switch(PceCalculation).join(Measurement).join(Parameter).join(Model)
                                   .where(Model.mtype == a.model and Model.dim == a.d and Model.basis == a.basis
                                                                 and Model.alpha == a.alpha and Model.noise == a.sigma)
                                   .where(Parameter.p_id == a.p)
                                   .where(Measurement.m_id == a.m)
                                   .where(Multiset.stype == a.stype)
                                   .where(PceCalculation.methd == a.appr)
                                   .where(PceEvaluation.methd == a.int))

    if (   (    a.stype == 'totalorder'
            and any(ev.calc.multi.order == a.order for ev in evals))
        or (    a.stype == 'anisotropic'
            and any(ev.calc.multi.alpha == a.alpha and ev.calc.multi.epsln == a.eps for ev in evals))):
        print(INFO + 'Nothing to do.')
        return

    modelDBO = Model.get_or_create_from_args(a)
    model    = modelDBO.get_model()
    paramDBO = Parameter.get_or_create_from_args(modelDBO, a)
    msrmtDBO = Measurement.get_or_create_from_args(modelDBO, model, paramDBO, a)


    print(INFO + 'Creating multiset...', end=' ')
    multiDBO = Multiset.get_or_create_from_args(a)
    print(' Done. ({:.6f}s)'.format(multiDBO.ctime))
    if multiDBO.size > 5000 :
        print(INFO + 'Skipping multiset of size ', multiDBO.size)
        if a.order : QUIT_ISO = True
        else :  QUIT_ANI = True
        return

    postr = GaussianPosterior(noise=modelDBO.noise, y_measurement=fr_string(msrmtDBO.y_val))
    graph = mm.WorkGraph()
    graph.AddNode(model, 'model')
    graph.AddNode(postr, 'postr')
    graph.AddEdge('model', 0, 'postr', 0)
    target = graph.CreateModPiece('postr')
    #target = SimpleTarget(a.d)

    sqrt = mm.SqrtOperator(1)
    graph = mm.WorkGraph()
    graph.AddNode(target, 'target')
    graph.AddNode(sqrt, 'sqrt')
    graph.AddEdge('target', 0, 'sqrt', 0)
    target = graph.CreateModPiece('sqrt')

    print(INFO + 'Creating PCE...', end=' ')
    pcecalcDBO = PceCalculation.get_or_create_from_args(msrmtDBO, multiDBO, target, a)
    print(' Done. ({:.6f}s)'.format(pcecalcDBO.ctime))

    print(INFO + 'Computing L2...', end=' ')
    pceevalDBO = PceEvaluation.get_or_create_from_args(pcecalcDBO, multiDBO, target, a)
    print(' Done. ({:.6f}s)'.format(pceevalDBO.ctime))

    return pceevalDBO


#-------------------------------------------------------------------------------
#    Main
#-------------------------------------------------------------------------------

if __name__ == "__main__" :

    args = create_args()
    print('\tArguments were ', args)

    start = datetime.datetime.now()

    order_list = [0,2,3,4,5,6,7,8,10,12,14,16,18,20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
    eps_list = [1/(10**i) for i in range(1,21)]#51)]

    try :
        with ProcessPoolExecutor(max_workers=args.cpu_count) as executor:

            if False :
                for d in [16] :
                    if QUIT : break
                    args.d = d
                    for alpha in [1,2,4,8] :
                        if QUIT : break
                        args.alpha = alpha

                        args.stype = 'totalorder'
                        for order in order_list :
                            if QUIT : break
                            args.order = order
                            executor.submit(create_data, deepcopy(args))
                            #create_data(deepcopy(args))

                        args.stype = 'anisotropic'
                        for eps in eps_list :
                            if QUIT : break
                            args.eps = eps
                            executor.submit(create_data, deepcopy(args))
                            #create_data(deepcopy(args))
            else :
                args.stype = 'totalorder'
                for order in order_list :
                    if QUIT : break
                    args.order = order
                    #executor.submit(create_data, deepcopy(args))
                    create_data(deepcopy(args))

                args.stype = 'anisotropic'
                for eps in eps_list :
                    if QUIT : break
                    args.eps = eps
                    #executor.submit(create_data, deepcopy(args))
                    create_data(deepcopy(args))

    except KeyboardInterrupt : QUIT = True

    elapsed = datetime.datetime.now() - start
    print('~~~> Runtime was ' + str(elapsed))


