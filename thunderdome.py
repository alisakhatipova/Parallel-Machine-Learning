import numpy as np
import math

from sklearn.metrics import roc_auc_score
from sklearn import datasets
from constants import *
from algorithms import *

mode = 'DEBUG'


class ThunderDome:
    def __init__(self, inputfile, alg):
        self.alg = alg
        if mode == 'DEBUG':
            cancer = datasets.load_breast_cancer()
            all_set = cancer.data
            classes = cancer.target.reshape((-1, 1))
            all_set, classes = self.shuffle_data(all_set, classes)
        else:
            X = np.loadtxt(inputfile, dtype='f4', delimiter=',')
            classes = X[:, 0].reshape((-1, 1))
            all_set = X[:, 1:]

        self.train_set, self.test_set = np.array_split(all_set, 2)
        self.train_classes, self.test_classes = np.array_split(classes, 2)
        self.models = []

    def split_data(self, data_set, classes, g):
        n = math.ceil(len(self.models)*1.0 / g)
        return np.array_split(data_set, n), np.array_split(classes, n)

    @staticmethod
    def shuffle_data(data, classes):
        np.random.seed(0)
        idx = np.random.permutation(len(data))
        return data[idx], classes[idx]

    @staticmethod
    def split_models(models, g):
        n = int(math.ceil(len(models) * 1.0 / g))
        k, m = divmod(len(models), n)
        return [models[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]

    def run(self):
        self.train()
        while True:
            groups = self.split_models(self.models, GROUP_SIZE)
            self.test_set, self.test_classes = self.shuffle_data(self.test_set, self.test_classes)
            test_sets, test_classes_sets = self.split_data(self.test_set, self.test_classes, GROUP_SIZE)
            winners = []
            for group, test_set, test_classes_set in zip(groups, test_sets, test_classes_sets):
                best_model = self.competition(group, test_set, test_classes_set)
                winners.append(best_model)
            self.models = winners
            if len(self.models) == 1:
                return self.models[0]

    def train(self):
        train_subsets = np.array_split(self.train_set, MODELS_NUM)
        train_classes = np.array_split(self.train_classes, MODELS_NUM)
        for train_subset, train_class in zip(train_subsets, train_classes):
            model = self.alg(train_subset, train_class)
            self.models.append(model)

    @staticmethod
    def competition(models, test_set, test_classes):
        best_model = None
        best_auc = 0.0
        for model in models:
            predicted_classes = model.predict(test_set)
            auc = roc_auc_score(test_classes, predicted_classes)
            if auc > best_auc:
                best_auc = auc
                best_model = model
        return best_model


if __name__ == "__main__":

    if dataset == 'HIGGS':
        input_file = 'datasets/HIGGS.csv'
    elif dataset == 'SUSY':
        input_file = 'datasets/SUSY.csv'

    experiment = ThunderDome(input_file, alg_svm)
    experiment.run()
