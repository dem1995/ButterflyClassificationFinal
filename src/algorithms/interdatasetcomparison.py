'''
Makes predictions about butterflies using the persistance image vectors.
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

import mystats

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
import normalitycheck as normalitycheck

import warnings


def runcomparisons():
	warnings.simplefilter(action='ignore', category=FutureWarning)
	warnings.simplefilter(action='ignore', category=UserWarning)
	#Whether to output the results to the console or to files
	CONSOLEOVERFILES = True
	SAVEIMAGES = True

	#Determine the directories for the features and targets
	cwd = getcwd()
	resultdir = f"{cwd}/results"
	storagestring = "Comparisons"

	#Determine the directories for the features and targets
	featuretabledir = f"{cwd}/featuretables"
	classifications = pd.read_csv(f"{featuretabledir}/classification.csv")
	together_bovw = classifications
	together_bovw_with_persims = classifications

	bovwfeatures = pd.read_csv(f"{featuretabledir}/{'bovw'}.csv")
	together_bovw = pd.merge(together_bovw, bovwfeatures)
	together_bovw_with_persims = pd.merge(together_bovw_with_persims, bovwfeatures)

	persimfeatures = pd.read_csv(f"{featuretabledir}/{'persims'}.csv")
	#together_bovw = pd.merge(together_bovw, persimfeatures)
	together_bovw_with_persims = pd.merge(together_bovw_with_persims, persimfeatures)

	df1 = together_bovw.drop(columns="Butterfly File")
	df2 = together_bovw_with_persims.drop(columns="Butterfly File")


	#Retrieve the parameters and create the classifiers

	with open(f"{resultdir}/paramspersims.pickle", 'rb') as paramfile:
		params_persim = pickle.load(paramfile)

	with open(f"{resultdir}/paramsbovw.pickle", 'rb') as paramfile:
		params_bovw = pickle.load(paramfile)

	with open(f"{resultdir}/paramsbovw_with_persim.pickle", 'rb') as paramfile:
		params_bovw_with_persim = pickle.load(paramfile)

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


	print("BOVW params:")
	print(params_bovw)
	print("Persim params:")
	print(params_persim)
	print("BOVW&Persim Params:")
	print(params_bovw_with_persim)


	classifier1s = {classifiertype:classifiertype(**params_bovw[classifiertype]) for classifiertype in classifiernames.keys()}
	classifier2s = {classifiertype:classifiertype(**params_bovw_with_persim[classifiertype]) for classifiertype in classifiernames.keys()}

	for classifiertype in classifiernames.keys():
		#print(tuple(pair))
		classifier1 = classifier1s[classifiertype]
		classifier2 = classifier2s[classifiertype]
		print(f"Comparing {classifiernames[classifiertype]}")
		results = mystats.run5x2pairedttest_differentsources(df1, df2, classifier1, classifier2)
		#null hypothesis of the two algorithms' performance being the same is rejected if p<alpha=0.05
		isstatisticallysignificant = (results.p<0.05)
		if isstatisticallysignificant:
			print(f"Stastically significant difference. p={results.p}")
		else:
			print(f"No stastically significant difference. p={results.p}")
		print(mystats.run5x2pairedttest_scores_differentsources(df1, df2, classifier1, classifier2))

if __name__ == '__main__':
	runcomparisons()