from paramselect import select_and_store_best_params_to_pickle
from evaluation import runtests
from interdatasetcomparison import runcomparisons

from enum import Enum

DATASET = Enum('Dataset', ['bovw', 'persims', 'bovw_with_persim'])

#Determine hyperparameters for each algorithm for each feature set
select_and_store_best_params_to_pickle()

#Run tests and do comparisons within a featureset
for enumval in DATASET:
	runtests(10, enumval, print_to_console_instead_of_file=True, saveimages=True)

#Run tests between featuresets
runcomparisons()