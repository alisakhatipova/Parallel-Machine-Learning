from thunderdome import *
from constants import *
from algorithms import *
import sys
import os
import itertools
from os import path

groups_sizes = [2, 5, 10, 20, 100]
models_number = [10, 100, 500, 1000]
if __name__ == "__main__":

    # f = open(path.join("error.log"), "w")
    # original_stderr = sys.stderr
    # sys.stderr = f

    if dataset == 'HIGGS':
        input_file = 'datasets/HIGGS.csv'
    elif dataset == 'SUSY':
        input_file = 'datasets/SUSY.csv'
    log_folder = 'base_experiment'

    directory = log_folder
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    experiment = ThunderDome(input_file, alg_svm, 2, 20, log_folder)
    experiment.run_experiment()
    experiment = ThunderDome(input_file, alg_svm, 4, 20, log_folder, False)
    experiment.run_experiment()

    experiment = ThunderDome(input_file, alg_knn, 2, 20, log_folder)
    experiment.run_experiment()
    experiment = ThunderDome(input_file, alg_knn, 4, 20, log_folder, False)
    experiment.run_experiment()

    # [zip(x, groups_sizes) for x in itertools.permutations(models_number, len(groups_sizes))]


