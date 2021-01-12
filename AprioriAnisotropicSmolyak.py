
import numpy as np
import pymuqModeling_ as mm
import pymuqUtilities as mu
import pymuqApproximation as ma

from GaussianPosterior import *

from Convolution import *
import basis

x_measurement = np.linspace(-1,1,10)
param  = [.1,.2,.4]
noise = .1
d = len(param)

conv = Convolution(basis=basis.hats, d=d, alpha=1, x_val=x_measurement)
y_measurement = conv.Evaluate([param]) + noise*np.random.randn(len(x_measurement))
post = GaussianPosterior(noise=noise, y_measurement=y_measurement)



d = 3

model = mm.CosOperator(d)
quad1d = ma.LejaQuadrature()
polys1d = ma.Legendre()
smolyPCE = ma.AdaptiveSmolyakPCE(post, [quad1d]*d, [polys1d]*d, 'interpolation');

# Start with a linear approximation
initialOrder = 1
multis = mu.MultiIndexFactory.CreateAnisotropic([.5,.1, .05], .01)

options = dict()
options['ShouldAdapt']  = 1    # After constructing an initial approximation with the terms in "multis", should we continue to adapt?
options['ErrorTol']     = 1e-4 # Stop when the estimated L2 error is below this value

pce = smolyPCE.Compute(multis, options);

smolyPCE.PrintInfo()

print('Number of Model Evaluations:')
print(smolyPCE.NumEvals())

print('\nEstimated L2 Error:')
print('%0.4e'%smolyPCE.Error())
