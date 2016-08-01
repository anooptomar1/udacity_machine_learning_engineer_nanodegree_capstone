
import numpy as np
from constants import *
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import os
import cPickle as pickle
import gzip


class SketchClassifierFactory(object):

    @staticmethod
    def create_classifier(classifier, filename):
        if classifier == ClassifierKeys.KNN:
            return KNeighborsClassifierSketchClassifier(filename=filename)
        elif classifier == ClassifierKeys.GaussianNaiveBayes:
            return GaussianNaiveBayesSketchClassifier(filename=filename)
        elif classifier == ClassifierKeys.LinearSVM:
            return LinearSVCSketchClassifier(filename=filename)
        elif classifier == ClassifierKeys.MultinomialNaiveBayes:
            return MultinomialNaiveBayesSketchClassifier(filename=filename)
        elif classifier == ClassifierKeys.SVM:
            return SVCSketchClassifier(filename=filename)
        elif classifier == ClassifierKeys.Best:
            return BestSketchClassifier(filename=filename)

        raise Exception("Unknown classifier {}".format(classifier))


class BaseSketchClassifier(object):

    def __init__(self, filename=None):
        self.label_encoder = preprocessing.LabelEncoder()
        self.clf = None
        self.filename = filename
        self.random_state = 56

        self._trained = False

        if self.filename is not None:
            self.restore(self.filename)

        if self.clf is None:
            self.clf = self.build_classifier()

    def fit(self, X, labels):
        y = self._encodeLabels(labels)
        X = np.asarray(X)

        self.clf.fit(X, y)

        self._trained = True

        if self.filename is not None:
            self.store(self.filename)

        return self

    @property
    def is_trained(self):
        return self._trained

    def store(self, filename):
        """
        reference: http://scikit-learn.org/stable/modules/model_persistence.html
        """
        with gzip.GzipFile(filename, 'w') as f:
            pickle.dump((self.label_encoder, self.clf), f)

        return self

    def restore(self, filename):
        """
        reference: http://scikit-learn.org/stable/modules/model_persistence.html
        """
        if not os.path.isfile(filename):
            return None

        with gzip.GzipFile(filename, 'r') as f:
            self.label_encoder, self.clf = pickle.load(f)

        return self

    def _encodeLabels(self, labels_words):
        self.label_encoder.fit(labels_words)
        return np.array(self.label_encoder.transform(labels_words), dtype=np.float32)

    def classify(self, X):
        X = np.asarray(X)
        labels_nums = self.clf.predict(X)
        labels_words = self.label_encoder.inverse_transform([int(x) for x in labels_nums])
        return labels_words, labels_nums

    def prob(self, X):
        X = np.asarray(X)
        labels_nums = self.clf.predict_proba(X)
        probabilities = [
            [(self.label_encoder.inverse_transform(int(x)), labels_nums_item[x]) for x in range(len(labels_nums_item))]
            for labels_nums_item in labels_nums]
        return probabilities, labels_nums

    def build_classifier(self):
        raise Exception("build_classifier not implemented")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "BaseClassifier"


class LinearSVCSketchClassifier(BaseSketchClassifier):
    """
    C : float, optional (default=1.0)
    Penalty parameter C of the error term.

    loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
    Specifies the loss function. 'hinge' is the standard SVM loss (used e.g. by the SVC class) while 'squared_hinge' is the square of the hinge loss.

    penalty : string, 'l1' or 'l2' (default='l2')
    Specifies the norm used in the penalization. The 'l2' penalty is the standard used in SVC. The 'l1' leads to coef_ vectors that are sparse.

    multi_class: string, 'ovr' or 'crammer_singer' (default='ovr') :
    Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.
    """

    def __init__(self, filename=None):
        self.C = 0.
        self.loss = ""

        BaseSketchClassifier.__init__(self, filename)

    def fit(self, X, labels):
        y = self._encodeLabels(labels)
        X = np.asarray(X)

        self.clf.fit(X, y)

        self.set_best_params()

        self.set_best_estimator()

        self._trained = True

        if self.filename is not None:
            self.store(self.filename)

        return self

    def set_best_estimator(self):
        if self.clf is None:
            return

        if "best_estimator_" not in dir(self.clf):
            print "set_best_bestimator returning, no property best_estimator_"
            return

        self.clf = self.clf.best_estimator_

        print "best_estimator_: {}".format(self.clf)

    def set_best_params(self):
        if self.clf is None:
            return

        if "best_params_" not in dir(self.clf):
            print "set_best_params returning, no property best_params_"
            return

        best_params = self.clf.best_params_

        if "loss" in best_params:
            self.loss = best_params["loss"]

        if "C" in best_params:
            self.C = best_params["C"]

        print "best_params_: {}".format(best_params)

    def build_classifier(self):
        C_range = np.logspace(-2, 10, 12)
        loss_functions = ['squared_hinge', 'hinge']

        param_grid = dict(loss=loss_functions, C=C_range)

        clf = GridSearchCV(
            LinearSVC(random_state=self.random_state),
            param_grid=param_grid,
            verbose=100
        )

        return clf

    def __repr__(self):
        return "{}_C-{}_loss-{}".format(
            "LinearSVCSketchClassifier",
            self.C,
            self.loss
        )


class SVCSketchClassifier(BaseSketchClassifier):
    """
    http://scikit-learn.org/stable/modules/svm.html#svm-classification
    http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    """

    def __init__(self, filename=None, decision_function_shape='ovr'):
        self.kernel = ""  # linear, poly, rbf, sigmoid
        self.C = 0.
        self.gamma = 0.  # kernal coefficent for "rbf", "poly" and "sigmoid", 'linear'
        self.decision_function_shape = decision_function_shape  # one-vs.-rest or one-vs.-one

        BaseSketchClassifier.__init__(self, filename)

    def fit(self, X, labels):
        y = self._encodeLabels(labels)
        X = np.asarray(X)

        self.clf.fit(X, y)

        self.set_best_params()

        self.set_best_estimator()

        self._trained = True

        if self.filename is not None:
            self.store(self.filename)

        return self

    def set_best_estimator(self):
        if self.clf is None:
            return

        if "best_estimator_" not in dir(self.clf):
            print "set_best_bestimator returning, no property best_estimator_"
            return

        self.clf = self.clf.best_estimator_

        print "best_estimator_: {}".format(self.clf)

    def set_best_params(self):
        if self.clf is None:
            return

        if "best_params_" not in dir(self.clf):
            print "set_best_params returning, no property best_params_"
            return

        best_params = self.clf.best_params_

        if "kernel" in best_params:
            self.kernel = best_params["kernel"]

        if "gamma" in best_params:
            self.gamma = best_params["gamma"]

        if "C" in best_params:
            self.C = best_params["C"]

        print "best_params_: {}".format(best_params)

    def build_classifier(self):
        """
        GridSearchCV
        http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
        """

        kernels = ['linear', 'rbf', 'poly']
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(kernel=kernels, gamma=gamma_range, C=C_range)
        clf = GridSearchCV(
            SVC(probability=True, random_state=self.random_state),
            param_grid=param_grid,
            n_jobs=-1
        )

        return clf

    def __repr__(self):
        return "{}_kernal-{}_C-{}_gamma-{}".format(
            "SVCSketchClassifier",
            self.kernel,
            self.C,
            self.gamma
        )


class KNeighborsClassifierSketchClassifier(BaseSketchClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    """

    def __init__(self, filename=None, n_neighbors=10):

        self.n_neighbors = n_neighbors
        self.weights = ""
        self.metric = ""
        self.algorithm = ""

        BaseSketchClassifier.__init__(self, filename)

    def fit(self, X, labels):
        y = self._encodeLabels(labels)
        X = np.asarray(X)

        self.clf.fit(X, y)

        self.set_best_params()

        self.set_best_estimator()

        self._trained = True

        if self.filename is not None:
            self.store(self.filename)

        return self

    def set_best_estimator(self):
        if self.clf is None:
            return

        if "best_estimator_" not in dir(self.clf):
            print "set_best_bestimator returning, no property best_estimator_"
            return

        self.clf = self.clf.best_estimator_

        print "best_estimator_: {}".format(self.clf)

    def set_best_params(self):
        if self.clf is None:
            return

        if "best_params_" not in dir(self.clf):
            print "set_best_params returning, no property best_params_"
            return

        best_params = self.clf.best_params_

        if "weights" in best_params:
            self.weights = best_params["weights"]

        if "algorithm" in best_params:
            self.algorithm = best_params["algorithm"]

        if "metric" in best_params:
            self.metric = best_params["metric"]

        if "n_neighbors" in best_params:
            self.n_neighbors = best_params["n_neighbors"]

        print "best_params_: {}".format(best_params)

    def build_classifier(self):
        """
        n_neighbors:
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

        weights:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        metric:
        string or DistanceMetric object (default = 'minkowski')
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.

        distance metrics:
        'euclidean'	EuclideanDistance sqrt(sum((x - y)^2))
        'manhattan' ManhattanDistance sum(|x - y|)
        'chebyshev' ChebyshevDistance sum(max(|x - y|))
        'minkowski'	MinkowskiDistance	p	sum(|x - y|^p)^(1/p)
        'wminkowski'	WMinkowskiDistance	p, w	sum(w * |x - y|^p)^(1/p)
        'seuclidean'	SEuclideanDistance	V	sqrt(sum((x - y)^2 / V))
        'mahalanobis'	MahalanobisDistance	V or VI	sqrt((x - y)' V^-1 (x - y))
        """

        """
        GridSearchCV
        http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
        """

        n_neighbors_values = [5, 10, 15, 20, 25]
        distance_metrics_values = ['minkowski', 'euclidean', 'manhattan']
        weights_values = ["uniform", "distance"]

        param_grid = dict(n_neighbors=n_neighbors_values,
                          weights=weights_values,
                          metric=distance_metrics_values)

        clf = GridSearchCV(
            neighbors.KNeighborsClassifier(n_jobs=-1),
            param_grid=param_grid,
            verbose=5
        )

        return clf

    def __repr__(self):
        return "{}_neighors-{}_weights-{}_metric-{}_algorithm-{}".format(
            "KNeighborsClassifier",
            self.n_neighbors,
            self.weights,
            self.metric,
            self.algorithm
        )


class MultinomialNaiveBayesSketchClassifier(BaseSketchClassifier):
    """
    http://scikit-learn.org/stable/modules/neighbors.html
    http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
    http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

    example:
    http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py
    """

    def __init__(self, filename):

        self.alpha = 0.1

        BaseSketchClassifier.__init__(self, filename)

    def build_classifier(self):
        return MultinomialNB(
            alpha=self.alpha
        )

    def __repr__(self):
        return "{}".format(
            "MultinomialNB"
        )


class GaussianNaiveBayesSketchClassifier(BaseSketchClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    """

    def __init__(self, filename):

        BaseSketchClassifier.__init__(self, filename)

    def build_classifier(self):
        return GaussianNB()

    def __repr__(self):
        return "{}".format(
            "GaussianNB"
        )


class BestSketchClassifier(BaseSketchClassifier):
    """
    Defined after model evaluation and optimisation
    """

    def __init__(self, filename=None, decision_function_shape='ovr'):

        BaseSketchClassifier.__init__(self, filename)

    def build_classifier(self):
        return SVC(
            probability=True,
            random_state=self.random_state,
            kernel="rbf",
            C=10.,
            gamma=10.
        )

    def __repr__(self):
        return "BEST {}".format(
            "SVCSketchClassifier"
        )
