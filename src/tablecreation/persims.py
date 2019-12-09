from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname

import pandas as pd
import numpy as np


def createpersimtable(imgdir=None, persimdir=None, resultdir=None):
	'''
	Creates a .csv file with (1 + the number of pixels in a persistence image) columns.
	The first column contains the filename of the butterfly whence the persistence image derives;
	the second to the last indexth, the pixel values in the persistence image.
	'''

	#Get the butterfly image directory, the directory containing the persistence images to process,
	#and the location to put the finished csv file when all's said and done
	cwd = getcwd()
	if imgdir is None or persimdir is None or resultdir is None:
		cwd = getcwd()
		imgdir = cwd + "/data/leedsbutterfly/images"
		persimdir = cwd + "/processing/persims"
		resultdir = cwd + "/featuretables"

	if not exists(resultdir):
		makedirs(resultdir)	

	resultfile = f"{resultdir}/persims.csv"

	#Get the names of the butterfly files and the butterfly indices (the prefix for each butterfly file name)
	butterflyfiles = [f for f in listdir(imgdir)]
	butterflyindices = [f[:-4] for f in butterflyfiles]

	#Get the persisance image vector for each butterfly image
	persimmatrices = [np.load(f"{persimdir}/{index}.npy") for index in butterflyindices]
	persimvectors = [persimmatrix.flatten() for persimmatrix in persimmatrices]

	#Generate labels for the pixel feature columns
	pixelnumbers = range(len(persimvectors[0]))
	pixellabels = [f"Pixel Number {i}" for i in pixelnumbers]

	#Save the butterfly image filenames and the persistence image vectors to a .csv file
	df = pd.DataFrame(persimvectors, columns = pixellabels)
	df.insert(0, "Butterfly File", butterflyfiles)
	df.to_csv(resultfile, index=False)

if __name__ == '__main__':
	createpersimtable()