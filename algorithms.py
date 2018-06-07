from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def alg_svm(train_subset, train_class):
    model = LinearSVC()
    model.fit(train_subset, train_class)
    return model


def alg_knn(train_subset, train_class):
    model = KNeighborsClassifier()
    model.fit(train_subset, train_class)
    return model
