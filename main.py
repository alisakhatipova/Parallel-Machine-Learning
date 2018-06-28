from os import path
from thunderdome import *
from constants import *
from algorithms import *


group_sizes = [10, 100, 1000, 10000]
models_numbers = [100, 1000, 10000, 100000]
inputfiles = ['datasets/SUSY.csv', 'datasets/HIGGS.csv']
# for DEBUG
# group_sizes = [3, 4, 30]
# models_numbers = [10, 20]
# inputfiles = ['datasets/SUSY.csv']
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
        X = np.loadtxt(inputfile, dtype='f4', delimiter=',')
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
