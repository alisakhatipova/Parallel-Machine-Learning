from os import path
from thunderdome import *
from constants import *
from algorithms import *


# group_sizes = [1, 100]
# models_numbers = [100, 1000]
# inputfiles = ['datasets/SUSY.csv', 'datasets/HIGGS.csv']
# for DEBUG
group_sizes = [5]
models_numbers = [1000]
inputfiles = ['datasets/SUSY.csv']
algorithms = [alg_svm, alg_lr, alg_nn]
log_folder = 'base_experiment'

if __name__ == "__main__":
    directory = log_folder

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    log_file = path.join(log_folder, 'full.txt')
    for inputfile in inputfiles:
        X = np.loadtxt('datasets/SUSY.csv', dtype='f4', delimiter=',')
        np.random.seed(34445)
        classes = X[:, 0].reshape((-1, 1))
        all_set = X[:, 1:]
        idx = np.random.permutation(len(all_set))
        all_set = all_set[idx]
        classes = classes[idx]
        X = [all_set, classes]
        for model_num in models_numbers:
            for group_size in group_sizes:
                if group_size >= models_numbers:
                    continue
                for alg in algorithms:
                    # train models (only need to do it once for all experiments
                    # compute result and time for decentralized case
                    experiment = ThunderDome(X, alg, group_size, model_num, log_file,
                                             split_type=SIMPLE_SPLIT, mode=MODE_REAL)
                    experiment.run_experiment()
                    experiment.compute_real_result()

                    # baseline SIMPLE SPLIT
                    experiment = ThunderDome(X, alg, model_num, model_num, log_file,
                                             split_type=SIMPLE_SPLIT, need_to_retrain=False, mode=MODE_REAL)
                    experiment.run_experiment()

                    # the same experiment for FULL_SPLIT
                    experiment = ThunderDome(X, alg, group_size, model_num, log_file,
                                             split_type=FULL_SPLIT, need_to_retrain=False, mode=MODE_REAL)
                    experiment.run_experiment()

                    # baseline FULL SPLIT
                    experiment = ThunderDome(X, alg, model_num, model_num, log_file,
                                             split_type=FULL_SPLIT, need_to_retrain=False, mode=MODE_REAL)
                    experiment.run_experiment()

                    # the same experiment for FULL_SPLIT
                    experiment = ThunderDome(X, alg, group_size, model_num, log_file,
                                             split_type=NO_SPLIT, need_to_retrain=False, mode=MODE_REAL)
                    experiment.run_experiment()

                    # baseline FULL SPLIT
                    experiment = ThunderDome(X, alg, model_num, model_num, log_file,
                                             split_type=NO_SPLIT, need_to_retrain=False, mode=MODE_REAL)
                    experiment.run_experiment()
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