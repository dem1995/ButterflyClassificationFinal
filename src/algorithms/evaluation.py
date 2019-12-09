'''
Makes predictions about butterflies using either bovw, persistence image vectors, or both
'''

from os import listdir, getcwd, makedirs
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from itertools import product
from itertools import chain, combinations
import pickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn import preprocessing
import seaborn as sns
from scipy import interp
from scipy.stats import shapiro, anderson
from scipy.stats import normaltest as dagostino

from statistics import mean
from statistics import stdev
import kfoldsmethods as kfoldsmethods
from collections import defaultdict, namedtuple
from enum import Enum
import normalitycheck

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def runtests(numruns=30, DATASETENUM=None, print_to_console_instead_of_file=True, saveimages=True):

	DATASET = DATASETENUM
	if DATASET is None:
		DATASET = Enum('Dataset', ['bovw', 'persims', 'bovw_with_persim'])
		DATASET = DATASET.bovw_with_persim


	NUMRUNS = numruns

	#Whether to output the results to the console or to files
	CONSOLEOVERFILES = True
	SAVEIMAGES = True

	#Determine the directories for the features and targets
	cwd = getcwd()
	resultdir = f"{cwd}/results"
	storagestring = DATASET.name
	if DATASET==DATASET.bovw:
		prettystring ="Bag of Visual Words Features"
	if DATASET==DATASET.persims:
		prettystring ="Persistance Image Features"
	if DATASET==DATASET.bovw_with_persim:
		prettystring ="BoVW and Persistance Image Features"
	print(storagestring)
	print(prettystring)

	#Determine the directories for the features and targets
	featuretabledir = f"{cwd}/featuretables"
	classifications = pd.read_csv(f"{featuretabledir}/classification.csv")
	together = classifications
	if DATASET in {DATASET.bovw, DATASET.bovw_with_persim}:
		bovwfeatures = pd.read_csv(f"{featuretabledir}/{'bovw'}.csv")
		together = pd.merge(together, bovwfeatures)
	if DATASET in {DATASET.persims, DATASET.bovw_with_persim}:
		persimfeatures = pd.read_csv(f"{featuretabledir}/{'persims'}.csv")
		together = pd.merge(together, persimfeatures)

	df = together.drop(columns="Butterfly File")


	#Retrieve the parameters and create the classifiers
	with open(f"{resultdir}/params{storagestring}.pickle", 'rb') as paramfile:
		params = pickle.load(paramfile)

	#Parameters
	classifiernames = {
		DummyClassifier: "Random Classifier",
		LogisticRegression: "Logistic Regression",
		SVC: "Support Vector Classifier",
		MultinomialNB: "Multinomial Naive Bayes",
		MLPClassifier: "Neural Net (Multilayer Perceptron)"
	}

	#Redetermine where to output the results
	resultdir = f"{resultdir}/{storagestring}"
	if not os.path.exists(resultdir):
		os.makedirs(resultdir)


	accuracies = defaultdict(lambda: list())
	meanaccuracies = dict()
	confusionmatrices = defaultdict(lambda: list())
	meanconfmatrices = dict()
	rocfprs = defaultdict(lambda: None)
	roctprs = defaultdict(lambda: None)
	rocaucs = defaultdict(lambda: None)

	for i in range(NUMRUNS):
		print(f"Run number {i}/{NUMRUNS}")
		#Create the classifiers
		classifiers = {classifiertype:classifiertype(**params[classifiertype]) for classifiertype in classifiernames.keys()}
		#Get the results of doing k-folds cross validation
		runresults = kfoldsmethods.meanreturndetailsofkfolds(df, classifiers.values(), numfolds=2)
		runaccuracies = runresults.accuracies
		runconfmatrices = runresults.confusionmatrices
		for classifiertype in classifiernames.keys():
			associatedclassifier = classifiers[classifiertype]
			accuracies[classifiertype].append(runaccuracies[associatedclassifier])
			confusionmatrices[classifiertype].append(runconfmatrices[associatedclassifier])
		
	for key in accuracies.keys():
		#meanaccuracies[key] = np.mean(accuracies[key], axis=0)
		print(np.array(confusionmatrices[key]).shape)
		meanconfmatrices[key] = np.mean(confusionmatrices[key], axis=0)
		print(meanconfmatrices[key].shape)

	#Check normality
	normalityoutfile = None if CONSOLEOVERFILES else f"{resultdir}/normalitytests.txt"
	normalitycheck.checknormality(classifiernames, accuracies, alpha=0.05, writeout = normalityoutfile)

	#Compare algorithm accuracies
	accuracyoutputfile = None if CONSOLEOVERFILES else f"{resultdir}/accuracies.txt"
	meanaccuracies = {classifiertype:np.mean(accuracies[classifiertype], axis=0) for classifiertype in classifiernames.keys()}
	stdaccuracies = {classifiertype:np.std(accuracies[classifiertype], axis=0) for classifiertype in classifiernames.keys()}

	outputfile = None
	print("Accuracies (Mean +- 1.96 STD)", file=outputfile)
	for classifiertype in classifiernames.keys():
		classifiername = classifiernames[classifiertype]
		mean = meanaccuracies[classifiertype]
		std = stdaccuracies[classifiertype]
		print(f"{classifiername}: {mean} +- {1.96*std}", file=outputfile)

	#Get confusion matrices
	#meancms = {classifiertype:np.mean(confusionmatrices[classifiertype], axis=0) for classifiertype in classifiernames.keys()}
	meancms = meanconfmatrices


	#fig.suptitle(f"Average Confusion Matrices for 2-folds Stratified Validation \n (Averaged over {NUMRUNS} using {prettystring})")
	for classifiertype in [classifierkey for classifierkey in classifiernames.keys() if classifierkey!=DummyClassifier]:
		fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
		classifiername = classifiernames[classifiertype]
		meancm = meancms[classifiertype]
		sns.heatmap(meancm, annot=True, fmt='.2f', xticklabels=range(10), yticklabels=range(10), ax=axs)
		axs.set(ylabel='Actual', xlabel='Predicted', title=f"{classifiername} Balanced Accuracy Confusion Matrix for 2-folds Stratified Validation \n (Averaged over {NUMRUNS} runs using {prettystring})")
		if SAVEIMAGES:
			plt.savefig(f"{resultdir}/{classifiertype.__name__}ConfMat.png")
			plt.show(block=False)
			plt.close()


	def uniquepairs(iterable):
		pairs = list(product(iterable, iterable))
		uniquepairs = {frozenset(pair) for pair in pairs if pair[0] != pair[1]}
		return uniquepairs

	for pair in uniquepairs(classifiernames.keys()):
		#print(tuple(pair))
		(classifier1, classifier2) = tuple(pair)
		print(f"Comparing {classifiernames[classifier1]} and {classifiernames[classifier2]}")
		results = kfoldsmethods.run5x2pairedttest(df, classifiers[classifier1], classifiers[classifier2])
		#null hypothesis of the two algorithms' performance being the same is rejected if p<alpha=0.05
		isstatisticallysignificant = (results.p<0.05)
		if isstatisticallysignificant:
			betteralgo = list(pair)[int(meanaccuracies[classifier1] < meanaccuracies[classifier2])]
			worsealgo = list(pair)[int(meanaccuracies[classifier1] > meanaccuracies[classifier2])]
			print(f"{classifiernames[betteralgo]} is statistically significantly better than {classifiernames[worsealgo]} with p={results.p}")
		else:
			print(f"No stastically significant difference. p={results.p}")