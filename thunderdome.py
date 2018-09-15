import numpy as np
import math
import os
import shutil
import time
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from constants import *


class ThunderDome:
    def __init__(self, X, alg, group_size, models_num, logger,
                 need_to_retrain=True, split_type=SIMPLE_SPLIT):
        self.logger = logger
        self.split_type = split_type
        logger.info("\n\n")
        logger.info("Experiment start " + alg.__name__ + " group size = " +
                    str(group_size) + " models number = " + str(models_num) +
                    " split type is " + SPL_STR[split_type] + " SPLIT")
        self.alg = alg
        positive = self.simple_shuffle(X[0])
        negative = self.simple_shuffle(X[1])
        # shuffle here
        self.train_time = 0.0
        self.models_num = models_num
        self.group_size = group_size
        self.need_to_retrain = need_to_retrain
        # set up validation set
        pos_subsets = np.array_split(positive, 10)
        neg_subsets = np.array_split(negative, 10)
        self.validation_set = np.concatenate((pos_subsets[-1], neg_subsets[-1]))
        pos_len = len(pos_subsets[-1])
        neg_len = len(neg_subsets[-1])
        self.validation_classes = np.concatenate((np.ones(pos_len), np.zeros(neg_len)))
        positive = np.concatenate(pos_subsets[0:-1])
        negative = np.concatenate(neg_subsets[0:-1])

        self.pos_train_set, self.pos_test_set = np.array_split(positive, 2)
        self.neg_train_set, self.neg_test_set = np.array_split(negative, 2)
        data, classes = self.stratified_split_by_n(self.pos_train_set, self.neg_train_set, 1)
        self.train_set, self.train_classes = data[0], classes[0]
        data, classes = self.stratified_split_by_n(self.pos_test_set, self.neg_test_set, 1)
        self.test_set, self.test_classes = data[0], classes[0]
        self.models = []
        self.best_auc = 0
        if self.split_type == FULL_SPLIT:
            part_num = self.calculate_nodes_num(models_num, group_size)
            self.full_split_data, self.full_split_classes = \
                self.stratified_split_by_n(self.pos_test_set, self.neg_test_set, part_num)

    def split_data(self, data_set, classes, g):
        n = math.ceil(len(self.models)*1.0 / g)
        return np.array_split(data_set, n), np.array_split(classes, n)

    def split_data_by_n(self, data_set, classes, n):
        return np.array_split(data_set, n), np.array_split(classes, n)

    def stratified_split(self, positive, negative, g):
        n = int(math.ceil(len(self.models) * 1.0 / g))
        return self.stratified_split_by_n(positive, negative, n)

    def stratified_split_by_n(self, positive, negative, n):
        positive = self.simple_shuffle(positive)
        negative = self.simple_shuffle(negative)
        pos = np.array_split(positive, n)
        neg = np.array_split(negative, n)
        all_data = []
        classes = []
        for i in range(n):
            cur_data = np.concatenate((pos[i], neg[i]))
            all_data.append(cur_data)
            pos_len = len(pos[i])
            neg_len = len(neg[i])
            cur_classes = np.concatenate((np.ones(pos_len), np.zeros(neg_len)))
            classes.append(cur_classes)
        return all_data, classes


    def next_part_of_full_split(self, g):
        n = int(math.ceil(len(self.models)*1.0 / g))
        current_data = self.full_split_data[:n]
        current_classes = self.full_split_classes[:n]
        self.full_split_data = self.full_split_data[n:]
        self.full_split_classes = self.full_split_classes[n:]
        return current_data, current_classes

    @staticmethod
    def calculate_nodes_num(num, g):
        sum = 0
        n = num
        while n > 1:
            n = int(math.ceil(n * 1.0 / g))
            sum += n
        return sum

    @staticmethod
    def shuffle_data(data, classes):
        idx = np.random.permutation(len(data))
        return data[idx], classes[idx]

    @staticmethod
    def simple_shuffle(data):
        idx = np.random.permutation(len(data))
        return data[idx]

    @staticmethod
    def split_models(models, g):
        n = int(math.ceil(len(models) * 1.0 / g))
        k, m = divmod(len(models), n)
        return [models[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]

    def run_experiment(self):
        time, winner = self.run()
        auc = self.validate(winner)
        self.logger.info('End of validation')
        return self.train_time + time, auc
        # self.compute_real_result()

    def run(self):
        if self.need_to_retrain:
            shutil.rmtree(MODELS_FOLDER)
            os.mkdir(MODELS_FOLDER)
            self.train_time = self.train()
            for i, model in enumerate(self.models):
                joblib.dump(model, os.path.join(MODELS_FOLDER, str(i) + '.pkl'))
            with open("traintime", 'w') as f:
                f.write(str(self.train_time))
        else:
            self.models = []
            models_list = sorted(os.listdir(MODELS_FOLDER), key=lambda st: int(st.split('.')[0]))
            for filename in models_list:
                self.models.append(joblib.load(os.path.join(MODELS_FOLDER, filename)))
            with open("traintime") as f:
                for line in f:
                    self.train_time = float(line)
        round_num = 0
        self.logger.info('Start rounds')
        rounds_time = 0.0
        while True:
            round_num += 1
            start = time.time()
            # self.logger.info("Round number " + str(round_num))
            # self.logger.info("Number of models in this round " + str(len(self.models)))
            groups = self.split_models(self.models, self.group_size)
            if self.split_type == SIMPLE_SPLIT:
                test_sets, test_classes_sets = self.stratified_split(self.pos_test_set, self.neg_test_set, self.group_size)
                # test_sets, test_classes_sets = self.split_data(self.test_set, self.test_classes, self.group_size)
            elif self.split_type == FULL_SPLIT:
                test_sets, test_classes_sets = self.next_part_of_full_split(self.group_size)
            winners = []
            if self.split_type == NO_SPLIT:
                for group in groups:
                    best_model = self.competition(group, self.test_set, self.test_classes)
                    winners.append(best_model)
            else:
                for group, test_set, test_classes_set in zip(groups, test_sets, test_classes_sets):
                    best_model = self.competition(group, test_set, test_classes_set)
                    winners.append(best_model)
            rounds_time += (time.time() - start) / len(self.models)
            self.models = winners
            if len(self.models) == 1:
                self.logger.info('End rounds in ' + str(time.time() - start) + ' seconds')
                return rounds_time, self.models[0]

    def train(self):
        # train_subsets = np.array_split(self.train_set, self.models_num)
        # train_classes = np.array_split(self.train_classes, self.models_num)
        train_subsets, train_classes = self.stratified_split_by_n(self.pos_train_set, self.neg_train_set, self.models_num)
        start = time.time()
        self.logger.info('Start decentralized training')
        for train_subset, train_class in zip(train_subsets, train_classes):
            model = self.alg(train_subset, train_class)
            self.models.append(model)
        self.logger.info('End decentralized training ' + str(time.time() - start)
                         + ' seconds, time per model ' + str((time.time() - start)/self.models_num) + ' seconds')
        return (time.time() - start)/self.models_num

    def validate(self, model):
        predicted_classes = model.predict(self.validation_set)
        auc = roc_auc_score(self.validation_classes, predicted_classes)
        self.logger.info('Result for validation set is ' + str(auc))
        # print 'Result for validation set is', auc
        return auc

    def compute_real_result(self):
        start = time.time()
        self.logger.info('\n\nStart centralized training')
        model = self.alg(self.train_set, self.train_classes)
        train_time = time.time() - start
        self.logger.info('End centralized training ' + str(train_time) + ' seconds')
        self.logger.info('Start centralized prediction')
        predicted_classes = model.predict(self.validation_set)
        auc = roc_auc_score(self.validation_classes, predicted_classes)
        self.logger.info('For centralized case result is ' + str(auc) + '\n')
        return train_time, auc
        # print 'For not distributed case result is ', auc

    def competition(self, models, test_set, test_classes):
        best_model = None
        best_auc = 0.0
        for model in models:
            predicted_classes = model.predict(test_set)
            auc = roc_auc_score(test_classes, predicted_classes)
            if auc > best_auc:
                best_auc = auc
                best_model = model
        # self.logger.warning('best auc ' + str(best_auc))
        self.best_auc = best_auc
        # print "best auc", best_auc
        return best_model
