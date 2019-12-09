'''Runs normality checks on a provided list of accuracies'''

from scipy.stats import shapiro, anderson
from scipy.stats import normaltest as dagostino

def checknormality(classifiernames, accuracies, alpha=0.05, writeout=None):
	'''
	Checks the normality of a list of accuracies using Shapiro-Wilk, Anderson-Darling, and D'agostino-Pearson tests.
	Please keep in mind that this should not be interpreted as saying the data is normal, only that it's not being ruled out, and therefore feasibly is.
	'''

	#If the output file is None, things will print to the console.
	if writeout is not None:
		outputfile = open(f"{writeout}", 'w')
	else:
		outputfile = None

	#https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
	print("Normality tests", file=outputfile)
	print(f"D'agostino-Pearson: Reject normality if pvalue<=α={alpha}", file=outputfile)
	for classifiertype in classifiernames.keys():
		print(f"Test on {classifiernames[classifiertype]}", file=outputfile)
		stat, pvalue = dagostino(accuracies[classifiertype])
		if pvalue > alpha:
			print(f'p={pvalue:.4g} > α={alpha}.\tFail to reject H0, that the accuracies are following a Gaussian distribution.', file=outputfile)
		else:
			print(f'p={pvalue:.4g} <= α={alpha}.\tReject H0, that the accuracies are following a Gaussian distribution.', file=outputfile)
	print("", file=outputfile)

	#https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test#Test_for_normality
	print("Anderson-Darling", file=outputfile)
	for classifiertype in classifiernames.keys():
		print(f"Test on {classifiernames[classifiertype]}", file=outputfile)
		result = anderson(accuracies[classifiertype])
		print('Statistic: %.3f' % result.statistic, file=outputfile)
		for i in range(len(result.critical_values))[1:]:
			sl, cv = result.significance_level[i], result.critical_values[i]
			stat = result.statistic
			if stat < cv:
				print(f'At α={sl/100:.3f}: we have our critical value {cv:.3g}>{stat:.3g};\tthe distribution of accuracies is feasibly normal (fail to reject H0)', file=outputfile)
			else:
				print(f'At α={sl/100:.3f}: we have our critical value {cv:.3g}<={stat:.3g};\tthe distribution of accuracies does not look normal (reject H0)', file=outputfile)
	print("", file=outputfile)

	#https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
	print(f"Shapiro-Wilk: Reject normality if pvalue<=α={alpha}", file=outputfile)
	for classifiertype in classifiernames.keys():
		print(f"Test on {classifiernames[classifiertype]}", file=outputfile)
		stat, pvalue = shapiro(accuracies[classifiertype])
		if pvalue > alpha:
			print(f'p={pvalue:.4g} > α={alpha}.\tFail to reject H0, that the accuracies are following a Gaussian distribution.', file=outputfile)
		else:
			print(f'p={pvalue:.4g} <= α={alpha}.\tReject H0, that the accuracies are following a Gaussian distribution.', file=outputfile)

	if outputfile is not None:
		outputfile.close()