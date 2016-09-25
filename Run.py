import pandas
import matplotlib.pyplot as pyplot
import numpy

from Perceptron import Perceptron
from matplotlib.colors import ListedColormap

data_frame = pandas.read_csv('https://archive.ics.uci.edu/ml/'
                             'machine-learning-databases/iris/iris.data', header=None)
data_frame.tail()
print(data_frame)

samples = data_frame.iloc[0:100, 4].values
print(samples)
samples = numpy.where(samples == 'Iris-setosa', -1, 1)
print(samples)

sepal_petal_matrix = data_frame.iloc[0:100, [0, 2]].values
print(sepal_petal_matrix)

pyplot.scatter(sepal_petal_matrix[:50, 0], sepal_petal_matrix[:50, 1], color='red', marker='o', label='setosa')
pyplot.scatter(sepal_petal_matrix[50:100, 0], sepal_petal_matrix[50:100, 1], color='blue', marker='x', label='versicolor')

pyplot.xlabel('petal length')
pyplot.ylabel('sepal length')
pyplot.legend(loc='upper left')
pyplot.show()

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(sepal_petal_matrix, samples)

pyplot.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
pyplot.xlabel('Epochs')
pyplot.ylabel('Number of misclassifications')
pyplot.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    color_map = ListedColormap(colors[:len(numpy.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution),
                              numpy.arange(x2_min, x2_max, resolution))

    print(xx1)
    print(xx2)

    Z = classifier.predict(numpy.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    pyplot.contourf(xx1, xx2, Z, alpha=0.4, cmap=color_map)
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(numpy.unique(y)):
        pyplot.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=color_map(idx), marker=markers[idx], label=cl)

plot_decision_regions(sepal_petal_matrix, samples, classifier=perceptron)
pyplot.xlabel('sepal length [cm]')
pyplot.ylabel('petal length [cm]')
pyplot.legend(loc='upper left')
pyplot.show()