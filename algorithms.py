from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def alg_svm(train_subset, train_class):
    model = LinearSVC()
    model.fit(train_subset, train_class.ravel())
    return model


def alg_knn(train_subset, train_class):
    model = KNeighborsClassifier()
    model.fit(train_subset, train_class.ravel())
    return model


def alg_lr(train_subset, train_class):
    model = LogisticRegression()
    model.fit(train_subset, train_class.ravel())
    return model


def alg_nn(train_subset, train_class):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                          hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(train_subset, train_class.ravel())
    return model
