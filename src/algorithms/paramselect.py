'''
Tunes hyperparameters for algorithms and stores them to a pickle file.
'''

import os
from os import getcwd
import pickle

import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def selectbestparamswithkfolds(df, classifiernames, parameter_spaces, numfolds=5):
	'''Selects the best parameters for the data using grid-search and the provided parameter spaces.'''
	#region select parameters
	bestparamvalues = dict()
	for classifiertype in classifiernames.keys():
		classifier = classifiertype()
		clf = GridSearchCV(classifier, parameter_spaces[classifiertype], n_jobs=-1, cv=numfolds)
		X = df.loc[:, df.columns != 'Category']
		y = np.array(df[['Category']]).ravel()
		clf.fit(X, y)
		bestparamvalues[classifiertype] = clf.best_params_

	return bestparamvalues


def select_and_store_best_params_to_pickle(tablesdir=None, classificationtable=None, bovwtable=None, persimstable=None, resultsdir=None):
	'''Selects the best parameters for the data using grid-search into pickles.'''
	#Determine the directories for the features and targets
	if tablesdir is None:
		cwd = getcwd()
		tablesdir = f"{cwd}/featuretables"
	if classificationtable is None:
		classificationtable = "classification.csv"
	if bovwtable is None:
		bovwtable = "bovw.csv"
	if persimstable is None:
		persimstable = "persims.csv"
	if resultsdir is None:
		resultsdir = resultdir = f"{cwd}/results"


	#Make the result directory if it doesn't already exist
	if not os.path.exists(resultdir):
		os.makedirs(resultdir)


	COMBINATIONS = [(False, True), (True, False), (True, True)]


	for usebovw, usepersims in COMBINATIONS:

		featuresincluded = "none"
		if usebovw and usepersims:
			featuresincluded = "bovw_with_persim"
		elif usebovw:
			featuresincluded = "bovw"
		elif usepersims:
			featuresincluded = "persims"

		print("Using feature: ", featuresincluded)

		#Determine the directories for the features and targets
		classifications = pd.read_csv(f"{tablesdir}/{classificationtable}")
		if usebovw:
			bovwfeatures = pd.read_csv(f"{tablesdir}/{bovwtable}")
		if usepersims:
			persimfeatures = pd.read_csv(f"{tablesdir}/{persimstable}")

		#Merge the features/targets together and drop their image identifier
		together = classifications
		if usebovw:
			together = pd.merge(together, bovwfeatures)
		if usepersims:
			together = pd.merge(together, persimfeatures)

		df = together.drop(columns="Butterfly File")

		#Parameters
		classifiernames = {
			DummyClassifier: "Random Classifier",
			LogisticRegression: "Logistic Regression",
			SVC: "Support Vector Classifier",
			MultinomialNB: "Multinomial Naive Bayes",
			MLPClassifier: "Neural Net (Multilayer Perceptron)"
		}

		paramspaces = {
			DummyClassifier:{
			},

			LogisticRegression:{
				"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],		#Smaller values indicate greater regularization
				"penalty": ['none', 'l1', 'l2'],				#The regularization penalty
				"solver": ['lbfgs', 'sag', 'saga'],				#The type of solver for LogisticRegression to use
				"max_iter": [2000]
			},

			SVC:{
				"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],		#Smaller values indicate greater regularization
				'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],	#The type of kernel to use
				'gamma': ['scale', 'auto']						#The coefficient for the kernel:
																#	1/ (n_features * X.var()) in the caes of scale.
																#	1 / n_features in the case of auto
			},

			MultinomialNB:{
				"alpha": [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],		#The smoothing parameter
				"fit_prior": [True, False]						#Whether to learn prior probabilities or not
																#	(false results in a uniform distribution)
			},

			MLPClassifier: {
				'hidden_layer_sizes': [(100, 100,), (50, 50, 50), (50, 100, 50), (100,)],
				'activation': ['tanh', 'relu'],
				'alpha': [0.0001, 0.05],
				'learning_rate': ['constant', 'adaptive'],
				'max_iter': [2000]
			}
		}


		bestparams = selectbestparamswithkfolds(df, classifiernames, paramspaces)
		print(bestparams)

		with open(f"{resultdir}/params{featuresincluded}.pickle", 'wb') as paramfile:
			pickle.dump(bestparams, paramfile, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f"{resultdir}/params{featuresincluded}.pickle", 'rb') as paramfile:
			readparams = pickle.load(paramfile)

		print(readparams)

if __name__ == '__main__':
	select_and_store_best_params_to_pickle()