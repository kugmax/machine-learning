import numpy
class Perceptron(object):

    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    Attributes
    -----------
    weights : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.f_learningRate = eta
        self.n_learningIterates = n_iter

    def fit(self, data, samples):
        """Fit training data.
        Parameters
        ----------
        data : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples
        is the number of samples and
        [ 25 ]Training Machine Learning Algorithms for Classification
        n_features is the number of features.
        samples : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        self.weights = numpy.zeros(1 + data.shape[1])
        self.errors_ = []
        for _ in range(self.n_learningIterates):
            errors = 0
            for xi, target in zip(data, samples):
                print("xi    = ", xi)
                print("target = ", target)

                update = self.f_learningRate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return numpy.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)