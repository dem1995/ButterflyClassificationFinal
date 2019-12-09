'''
Computes persistence diagrams for the filtered images (specifically, those for the H0 and H1 homology groups)
Accomplishes this by taking the black pixels of each image to be point cloud elements, then generating the persistence diagram for that point cloud.
For point clouds of count greater than 3000, randomly samples 90% of those points (or 3000, if the quantity of 90% of those points is less than that)
	for use in computing the persistence diagram to avoid overlong computation.
'''

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname
from random import sample
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt

from ripser import Rips

def createpersdgmsfrombinarizedimages(imgdir, persdgmdir):
	'''
	imgdir is the absolute filepath of the directory to get images from.
	persdgmdir is the directory to store the results to. If images have already been processed into that directory
		(i.e., if persdgmdir/{proposedpersdgmname.png} already exists) the program will just ignore it
	Default subdirectories of cwd (which itself defaults to using getcwd()) are used if these are None.
	'''

	rips = Rips()

	#make a directory to store the persistence diagrams in
	if not exists(persdgmdir):
		makedirs(persdgmdir)

	butterflyindices = [f[0:-10] for f in listdir(imgdir) if ("thresh" in f)]
	processedindices = [f[0:-4] for f in listdir(persdgmdir) if ("png" in f)]

	#Compute the persistence diagrams
	starttime = time.time()
	i=0
	for butterflyindex in butterflyindices:
		i+=1
		if not butterflyindex in processedindices:
			print(f"Processing butterfly {i}: {butterflyindex}")

			#Get the image
			image = cv2.imread(imgdir+"/"+butterflyindex+"thresh.png", 0)

			#Get the coordinates of the black pixels
			data = list()
			for row in range(image.shape[0]):
				for column in range(image.shape[1]):
					#print(image[row][column])
					if image[row][column]!=255:
						data.append((row, column))
						#print(f"{row}, {column}")

			#Randomly sample points if the size of the image is going to make things take too long
			minsamples = 3000
			if len(data)>minsamples:
				numsamples = minsamples if int(0.9*len(data))<minsamples else int(0.9*len(data))
				data = sample(data, numsamples)

			print(len(data))

			#Compute the persistence diagrams
			data = np.array(data)
			dgms = rips.fit_transform(data)

			#Plot the persistence diagram and save the results to a file
			plt.figure(figsize=(10,5))
			plt.subplot(121)
			plt.scatter(data[:,0], data[:,1], s=4)
			plt.title(f"Plot of sampled points of butterfly {butterflyindex}")

			plt.subplot(122)
			rips.plot(dgms, legend=False, show=False)
			plt.title("Persistence diagram of $H_0$ and $H_1$")

			plt.savefig(f"{persdgmdir}/{butterflyindex}.png")
			plt.close()

			#Save the persistence diagrams's array representation for usage by persimcreate.py
			np.save(f"{persdgmdir}/{butterflyindex}persdiag", dgms)

			#Print how much time the process took for a given butterfly image
			endtime = time.time()
			print(endtime-starttime, " seconds")
			starttime=endtime
		else:
			print(f"Butterfly {i}: {butterflyindex} already processed")
