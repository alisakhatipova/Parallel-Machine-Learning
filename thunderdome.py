import numpy as np
import math
import os
import logging
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn import datasets


class ThunderDome:
    def __init__(self, inputfile, alg, group_size, models_num, log_folder, need_to_retrain=True, mode='DEBUG'):
        # logging.basicConfig(filename='log.txt', level=logging.INFO,
        #                     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        # logger = logging.getLogger(__name__)
        self.alg = alg
        if mode == 'DEBUG':
            all_set, classes = datasets.load_breast_cancer(return_X_y=True)
            all_set, classes = self.shuffle_data(all_set, classes)
        else:
            X = np.loadtxt(inputfile, dtype='f4', delimiter=',')
            classes = X[:, 0].reshape((-1, 1))
            all_set = X[:, 1:]

        self.log_folder = log_folder
        self.models_num = models_num
        self.group_size = group_size
        self.need_to_retrain = need_to_retrain
        # set up validation set
        subsets = np.array_split(all_set, 10)
        self.validation_set = subsets[-1]
        all_set = np.concatenate(subsets[0:-1])
        subclasses = np.array_split(classes, 10)
        self.validation_classes = subclasses[-1]
        classes = np.concatenate(subclasses[0:-1])
        # set up train and test sets
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

    def run_experiment(self):
        winner = self.run()
        self.validate(winner)
        self.compute_real_result()

    def run(self):
        if self.need_to_retrain:
            self.train()
            for i, model in enumerate(self.models):
                joblib.dump(model, os.path.join('models', 'model' + str(i) + '.pkl'))
        else:
            self.models = []
            for filename in os.listdir('models'):
                self.models.append(joblib.load(os.path.join('models', filename)))
        round_num = 0
        while True:
            round_num += 1
            print "Round number", round_num, '\n'
            print "Number of models in this round", len(self.models), '\n'
            groups = self.split_models(self.models, self.group_size)
            self.test_set, self.test_classes = self.shuffle_data(self.test_set, self.test_classes)
            test_sets, test_classes_sets = self.split_data(self.test_set, self.test_classes, self.group_size)
            winners = []
            for group, test_set, test_classes_set in zip(groups, test_sets, test_classes_sets):
                best_model = self.competition(group, test_set, test_classes_set)
                winners.append(best_model)
            self.models = winners
            if len(self.models) == 1:
                return self.models[0]

    def train(self):
        train_subsets = np.array_split(self.train_set, self.models_num)
        train_classes = np.array_split(self.train_classes, self.models_num)
        for train_subset, train_class in zip(train_subsets, train_classes):
            model = self.alg(train_subset, train_class)
            self.models.append(model)

    def validate(self, model):
        predicted_classes = model.predict(self.validation_set)
        auc = roc_auc_score(self.validation_classes, predicted_classes)
        print 'Result for validation set is', auc
        return auc

    def compute_real_result(self):
        model = self.alg(self.train_set, self.train_classes)
        predicted_classes = model.predict(self.test_set)
        auc = roc_auc_score(self.test_classes, predicted_classes)
        print 'For not distributed case result is ', auc

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
        print "best auc", best_auc, '\n'
        return best_model
