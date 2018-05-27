from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def alg_svm(train_subset, train_class):
    model = SVC()
    model.fit(train_subset, train_class)
    return model


def alg_knn(train_subset, train_class):
    model = KNeighborsClassifier()
    model.fit(train_subset, train_class)
    return model
