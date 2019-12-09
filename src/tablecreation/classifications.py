from os import getcwd, makedirs, listdir
from os.path import exists

import pandas as pd

def createclassificationtable(imgdir=None, resultdir=None):
	'''
	Creates a .csv file with two columns: one for the butterfly image filenames; the other, for their targets (category of butterfly)
	'''
	if imgdir is None or resultdir is None:
		#Get the directories for the butterfly images and for that for which to output the .csv file
		cwd = getcwd()
		imgdir = cwd + "/data/leedsbutterfly/images"
		resultdir = cwd + "/featuretables"

	if not exists(resultdir):
		makedirs(resultdir)

	resultfile = f"{resultdir}/classification.csv"

	butterflyindices = [f for f in listdir(imgdir) if 'png' in f]

	#Make a pandas dataframe with the two columns' information and output it to a file
	classificationrows = [ [index, int(index[0:3])] for index in butterflyindices]
	df = pd.DataFrame(classificationrows, columns = ['Butterfly File', 'Category'])
	df.to_csv(resultfile, index=False)

if __name__ == '__main__':
	createclassificationtable()