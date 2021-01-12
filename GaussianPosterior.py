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

if __name__ == '__main__' :

    post = GaussianPosterior(noise=.1, y_measurement=np.random.uniform(size=(40,)))
    post.Evaluate([np.random.uniform(size=(40,))])
