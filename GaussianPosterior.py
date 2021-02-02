import numpy as np
import pymuqModeling_ as mm

class GaussianPosterior(mm.PyModPiece):

    def __init__(self, *, noise, y_measurement):
        super(GaussianPosterior, self).__init__([len(y_measurement)], [1])
        self.noise = noise
        self.y_measurement = y_measurement

    def EvaluateImpl(self, inputs):
        diff = np.subtract(self.y_measurement, inputs[0])
        self.outputs = [np.array([np.exp(-np.dot(diff.T, diff).item()/(2*self.noise**2))])]

class GaussianPosteriorSqrt(mm.PyModPiece):

    def __init__(self, *, noise, y_measurement):
        super(GaussianPosterior, self).__init__([len(y_measurement)], [1])
        self.noise = noise
        self.y_measurement = y_measurement

    def EvaluateImpl(self, inputs):
        diff = np.subtract(self.y_measurement, inputs[0])
        self.outputs = [np.array([np.sqrt(np.exp(-np.dot(diff.T, diff).item()/(2*self.noise**2)))])]

class GaussianMixture(mm.PyModPiece) :

    def __init__(self, *, noiseLevels, measurementList):
        super(GaussianMixture, self).__init__([len(measurementList[0])], [1])
        assert(len(noiseLevels) == len(measurementList))
        self.gaussians = [GaussianPosterior(noise=n, y_measurement=y) for n, y in zip(noiseLevels, measurementList)]

    def EvaluateImpl(self, inputs):
        self.outputs = [np.array([np.sum([g.Evaluate(inputs)[0] for g in self.gaussians])])]

if __name__ == '__main__' :

    #post = GaussianPosterior(noise=.1, y_measurement=np.random.uniform(size=(40,)))
    #post.Evaluate([np.random.uniform(size=(40,))])

    target = GaussianMixture(noiseLevels=[.1, .02], measurementList=[[-.5], [.5]])
