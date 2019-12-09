from collections import namedtuple, defaultdict
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets, metrics, model_selection
from mlxtend.evaluate import paired_ttest_5x2cv
import numpy as np
import partitioning



def run5x2pairedttest(df, classifier1, classifier2, ycolumnname='Category'):
	Pairedt5x2Details = namedtuple("Pairedt5x2Details", "t p")
	X = df.loc[:, df.columns!=ycolumnname]
	y = np.array(df[[ycolumnname]]).ravel()
	t, p = paired_ttest_5x2cv(estimator1=classifier1, estimator2=classifier2, X=X, y=y)

	return Pairedt5x2Details(t, p)

def returndetailsofkfolds(df, classifiers, numfolds=5, partitions=None):
	KFoldsDetails = namedtuple("KFoldsDetails", "accuracies confusionmatrices")
	accuracies = defaultdict(lambda: list())
	confusionmatrices = defaultdict(lambda: list())

	if partitions is None:
		partitions = partitioning.kfolds_partition(df, numfolds=numfolds, ycolumnname='Category', stratification=True)

	for training, testing in partitions:
		#Get the X and y matrices/vectors
		trainingx = training.drop(columns=['Category'], axis=1)
		trainingy = np.array(training.pop('Category').to_frame()).ravel()
		testingx = testing.drop(columns=['Category'], axis=1)
		testingy = np.array(testing.pop('Category').to_frame()).ravel()
		for classifier in classifiers:
			#Fit and make predictions
			classifier.fit(X=trainingx, y=trainingy)
			yhat = classifier.predict(testingx)
			#Store accuracy
			accuracies[classifier].append(balanced_accuracy_score(testingy, yhat))
			#Normalize and store confusion matrix
			unnormalizedcm = confusion_matrix(testingy, yhat)
			normalizedcm = unnormalizedcm/unnormalizedcm.sum(axis=1, keepdims=True)
			confusionmatrices[classifier].append(normalizedcm)

	return KFoldsDetails(accuracies=accuracies, confusionmatrices=confusionmatrices)

def meanreturndetailsofkfolds(df, classifiers, numfolds=5, partitions=None):
	KFoldsDetails = namedtuple("KFoldsDetails", "accuracies confusionmatrices")

	kfoldsdetails = returndetailsofkfolds(df, classifiers, numfolds, partitions)
	accuracies = kfoldsdetails.accuracies
	confusionmatrices = kfoldsdetails.confusionmatrices
	for classifier in classifiers:
		#print("3333333333333333333333333333333333---------------------------")
		#print(np.array(confusionmatrices[classifier]).shape)
		#print("4444444444444444444444444444444444---------------------------")
		#print(np.array(confusionmatrices[classifier]))
		accuracies[classifier] = np.mean(accuracies[classifier])
		confusionmatrices[classifier] = np.mean(confusionmatrices[classifier], axis=0)

	return KFoldsDetails(accuracies=accuracies, confusionmatrices=confusionmatrices)
