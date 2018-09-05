from os import path
from thunderdome import *
from constants import *
from algorithms import *
from sklearn import datasets
import matplotlib.pyplot as plt
import logging

algorithms = [alg_svm, alg_lr, alg_nn]


def split_comparison_experiment(X):
    group_size = 2
    model_num = 10
    res_full = {alg: [] for alg in algorithms}
    res_full_baseline = {alg: [] for alg in algorithms}
    res_simple = {alg: [] for alg in algorithms}
    res_simple_baseline = {alg: [] for alg in algorithms}
    res_no = {alg: [] for alg in algorithms}
    res_no_baseline = {alg: [] for alg in algorithms}
    res_real = {alg: [] for alg in algorithms}
    train_time_real = {alg: [] for alg in algorithms}
    train_time_decentr = {alg: [] for alg in algorithms}
    for i in range(iter_num):
        logger.info("ITERATION NUMBER " + str(i) + ' ***************************************************************')
        for alg in algorithms:
            # train models (only need to do it once for all experiments
            # compute result and time for decentralized case
            experiment = ThunderDome(X, alg, group_size, model_num, logger,
                                     split_type=SIMPLE_SPLIT)
            tr_time, auc = experiment.run_experiment()
            train_time_decentr[alg].append(time)
            res_simple[alg].append(auc)
            tr_time, auc = experiment.compute_real_result()
            train_time_real[alg].append(time)
            res_real[alg].append(auc)

            # baseline SIMPLE SPLIT
            experiment = ThunderDome(X, alg, model_num, model_num, logger,
                                     split_type=SIMPLE_SPLIT, need_to_retrain=False)
            _, auc = experiment.run_experiment()
            res_simple_baseline.append(auc)

            # the same experiment for FULL_SPLIT
            experiment = ThunderDome(X, alg, group_size, model_num, logger,
                                     split_type=FULL_SPLIT, need_to_retrain=False)
            _, auc = experiment.run_experiment()
            res_full.append(auc)

            # baseline FULL SPLIT
            experiment = ThunderDome(X, alg, model_num, model_num, logger,
                                     split_type=FULL_SPLIT, need_to_retrain=False)
            _, auc = experiment.run_experiment()
            res_full_baseline.append(auc)

            # the same experiment for NO_SPLIT
            experiment = ThunderDome(X, alg, group_size, model_num, logger,
                                     split_type=NO_SPLIT, need_to_retrain=False)
            _, auc = experiment.run_experiment()
            res_no.append(auc)

            # baseline NO SPLIT
            experiment = ThunderDome(X, alg, model_num, model_num, logger,
                                     split_type=NO_SPLIT, need_to_retrain=False)
            _, auc = experiment.run_experiment()
            res_no_baseline.append(auc)

    def group_size_comparison_experiment():
        group_sizes = [2, 5, 10, 20, 100]
        model_num = 1000
        for i in range(iter_num):
            logger.info(
                "ITERATION NUMBER " + str(i) + ' ***************************************************************')
            for group_size in group_sizes:
                for alg in algorithms:
                    # train models (only need to do it once for all experiments
                    # compute result and time for decentralized case
                    experiment = ThunderDome(X, alg, group_size, model_num, logger,
                                             split_type=SIMPLE_SPLIT)
                    experiment.run_experiment()
                    experiment.compute_real_result()

                    # baseline SIMPLE SPLIT
                    experiment = ThunderDome(X, alg, model_num, model_num, logger,
                                             split_type=SIMPLE_SPLIT, need_to_retrain=False)
                    experiment.run_experiment()

    def learners_num_comparison_experiment():
        models_nums = [2, 5, 10, 20, 100]
        for i in range(iter_num):
            logger.info(
                "ITERATION NUMBER " + str(i) + ' ***************************************************************')
            for model_num in models_nums:
                for alg in algorithms:
                    # train models (only need to do it once for all experiments
                    # compute result and time for decentralized case
                    experiment = ThunderDome(X, alg, group_size, model_num, logger,
                                             split_type=SIMPLE_SPLIT)
                    experiment.run_experiment()
                    experiment.compute_real_result()

                    # baseline SIMPLE SPLIT
                    experiment = ThunderDome(X, alg, model_num, model_num, logger,
                                             split_type=SIMPLE_SPLIT, need_to_retrain=False)
                    experiment.run_experiment()

if __name__ == "__main__":
    try:
        os.stat(LOG_FOLDER)
    except:
        os.mkdir(LOG_FOLDER)

    try:
        os.stat(MODELS_FOLDER)
    except:
        os.mkdir(MODELS_FOLDER)

    log_file = path.join(LOG_FOLDER, 'log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    np.random.seed(0)
    logger = logging.getLogger()
    # X = np.loadtxt('datasets/SUSY.csv', dtype='f4', delimiter=',')
    # classes = X[:, 0].reshape((-1, 1))
    # all_set = X[:, 1:]
    # idx = np.random.permutation(len(all_set))
    # all_set = all_set[idx]
    # classes = classes[idx]
    all_set, classes = datasets.load_breast_cancer(return_X_y=True)
    # all_set, classes = self.shuffle_data(all_set, classes)
    X = [all_set, classes]
    split_comparison_experiment(X)



    # X = np.loadtxt('datasets/SUSY.csv', dtype='f4', delimiter=',')
    # np.random.seed(34445)
    # classes = X[:, 0].reshape((-1, 1))
    # all_set = X[:, 1:]
    # idx = np.random.permutation(len(all_set))
    # all_set = all_set[idx]
    # classes = classes[idx]
    # all_set = np.array_split(all_set, 10)[0]
    # classes = np.array_split(classes, 10)[0]
    # np.savetxt('datasets/data10.csv', all_set, delimiter=',')
    # np.savetxt('datasets/classes10.csv', classes,  delimiter=',')
    # all_set = np.array_split(all_set, 1000)[0]
    # classes = np.array_split(classes, 1000)[0]