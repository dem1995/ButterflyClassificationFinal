'''Contains methods for partitioning data'''

import random
from collections import namedtuple
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

Partition = namedtuple('Partition', 'training testing')

def kfolds_partition(df, numfolds, ycolumnname='Category', stratification=False):
	'''
	Returns k-fold partitions of the given dataframe with the specified target column name
	The target column is mostly used for stratification, but because of how I wrote this, it's required regardless.
	'''
	if stratification:
		kfolder = StratifiedKFold(n_splits=numfolds, shuffle=True)
	else:
		kfolder = KFold(n_splits=numfolds, shuffle=True)

	x = df.drop(columns=[ycolumnname], axis=1)
	y = df[ycolumnname].to_frame()

	partitions = list()
	for train_indices, test_indices in kfolder.split(x, y):
		training = df.iloc[train_indices]
		testing = df.iloc[test_indices]
		partition = Partition(training=training, testing=testing)
		partitions.append(partition)

	return partitions

#Take df, which is a bunch of samples, and compute a training partition
#and a testing partition for each run. Training partitions are selected with repetition
def bootstrap_partitions(df, numruns, trainingsize):
	numsamples = df.shape[0]
	if trainingsize<1:
		trainingsize = int(trainingsize*numsamples)

	trainparts = list()
	for run in range(numruns):
		indices = [random.randint(0, numsamples-1) for i in range(trainingsize)]
		trainparts.append(indices)

	testparts = list()
	for run in range(numruns):
		indices = [index for index in range(numsamples) if index not in trainparts[run]]
		testparts.append(indices)

	partitions = list()
	for run in range(numruns):
		training = df.iloc[trainparts[run]]
		testing = df.iloc[testparts[run]]
		partition = Partition(training=training, testing=testing)
		partitions.append(partition)

	return partitions