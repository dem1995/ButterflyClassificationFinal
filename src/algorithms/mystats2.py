# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer


def paired_ttest_5x2cv(estimator1, estimator2, X1, X2, y,
					   scoring=None,
					   random_seed=None):
	"""
	Implements the 5x2cv paired t test proposed
	by Dieterrich (1998)
	to compare the performance of two models.

	Parameters
	----------
	estimator1 : scikit-learn classifier or regressor

	estimator2 : scikit-learn classifier or regressor

	X : {array-like, sparse matrix}, shape = [n_samples, n_features]
		Training vectors, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape = [n_samples]
		Target values.

	scoring : str, callable, or None (default: None)
		If None (default), uses 'accuracy' for sklearn classifiers
		and 'r2' for sklearn regressors.
		If str, uses a sklearn scoring metric string identifier, for example
		{accuracy, f1, precision, recall, roc_auc} for classifiers,
		{'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
		'median_absolute_error', 'r2'} for regressors.
		If a callable object or function is provided, it has to be conform with
		sklearn's signature ``scorer(estimator, X, y)``; see
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
		for more information.

	random_seed : int or None (default: None)
		Random seed for creating the test/train splits.

	Returns
	----------
	t : float
		The t-statistic

	pvalue : float
		Two-tailed p-value.
		If the chosen significance level is larger
		than the p-value, we reject the null hypothesis
		and accept that there are significant differences
		in the two compared models.

	Examples
	-----------
	For usage examples, please see
	http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/

	"""
	rng = np.random.RandomState(random_seed)

	if scoring is None:
		if estimator1._estimator_type == 'classifier':
			scoring = 'accuracy'
		elif estimator1._estimator_type == 'regressor':
			scoring = 'r2'
		else:
			raise AttributeError('Estimator must '
								 'be a Classifier or Regressor.')
	if isinstance(scoring, str):
		scorer = get_scorer(scoring)
	else:
		scorer = scoring

	variance_sum = 0.
	first_diff = None

	def score_diff(Xtrain1, Xtest1, Xtrain2, Xtest2, y_train, y_test):
		print(Xtrain1.shape)
		print(Xtrain2.shape)
		estimator1.fit(Xtrain1, y_train)
		estimator2.fit(Xtrain2, y_train)
		est1_score = scorer(estimator1, Xtest1, y_test)
		est2_score = scorer(estimator2, Xtest2, y_test)
		score_diff = est1_score - est2_score
		return score_diff

	for i in range(5):

		randint = rng.randint(low=0, high=32767)

		kfolder = KFold(n_splits=2)


		
		Xtrain1, Xtrain2, Xtest1, Xtest2, ytrain, ytest = ()
		#X_1, X_2, y_1, y_2 = \
		#	 train_test_split(X, y, test_size=0.5,
		#					  random_state=randint)

		# score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
		score_diff_1 = score_diff(Xtrain1, Xtrain2, Xtest1, Xtest2, ytrain, ytest)
		score_diff_2 = score_diff(Xtest1, Xtest2, Xtrain1, Xtrain2, ytest, ytrain)
		score_mean = (score_diff_1 + score_diff_2) / 2.
		score_var = ((score_diff_1 - score_mean)**2 +
					 (score_diff_2 - score_mean)**2)
		variance_sum += score_var
		if first_diff is None:
			first_diff = score_diff_1

	numerator = first_diff
	denominator = np.sqrt(1/5. * variance_sum)
	t_stat = numerator / denominator

	pvalue = stats.t.sf(np.abs(t_stat), 5)*2.
	return float(t_stat), float(pvalue)
