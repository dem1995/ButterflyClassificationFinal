"""
Calculates the persistence image (specifically, the persistence image for the H1 homology class) given persistence diagrams
"""

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from persim import PersImage


def createpersimsfrompersdgms(persdgmdir, persimdir):
	"""
	Calculates the persistence image (specifically, the persistence image for the H1 homology class) given persistence diagrams
	persdgmdir is the absolute filepath of the directory to get persistence diagram numpy vectors from (stored as an array  of length-2 vectors,
		the two elements of each of which represent the birth and death of a topological feature
	persimdir is the absolute filepath to which to store the processed persistence images. If images have already been processed into that directory
		(i.e., if persimdir/{proposedpersimname}.png already exists) the program will just ignore it
	Default subdirectories of cwd (which itself defaults to using getcwd()) are used if these are None.
	"""

	#make a directory to store the persistance images in
	if not exists(persimdir):
		makedirs(persimdir)

	butterflyindices = [f[0:-4] for f in listdir(persdgmdir) if (not "persdiag" in f)]
	processedindices = [f[0:-4] for f in listdir(persimdir) if ("png" in f)]

	#Retrieve the H1 persistence images and write them out to a file
	i = 0
	for butterflyindex in butterflyindices:
		i += 1
		if not butterflyindex in processedindices:
			print(f"Processing butterfly {i}/{len(butterflyindices)}. Index reference: {butterflyindex}")
			#Load the previously-saved persistence diagram
			filename = f"{persdgmdir}/{butterflyindex}persdiag.npy"
			dgms = np.load(filename, allow_pickle=True)

			#Calculate a persistence image from the H1 cycls of that diagram, then save it to a file
			pim = PersImage(spread=1, pixels=[10,10], verbose=False)
			img = pim.transform(dgms[1])
			np.save(f"{persimdir}/{butterflyindex}.npy", img)

			#Save a graphical representation of the persistence image as well
			plt.title(f"$H_1$ Persistence Image with 10x10 Pixel Resolution \n for Butterfly {butterflyindex}")
			pim.show(img)
			plt.savefig(f"{persimdir}/{butterflyindex}.png")
		else:
			print(f"Butterfly {i}/{len(butterflyindices)}: {butterflyindex} already processed")
