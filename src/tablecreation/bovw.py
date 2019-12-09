from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans as KMeans

def createbovwtable(imgdir=None, maskeddir=None, resultdir=None):
	'''
	Creates a .csv file with (1 + the number of features in the bag of visual words) columns.
	The first column contains the filename of the butterfly whence the persistence image derives;
	the second to the last indexth, the bag-of-visual-words features
	'''

	if imgdir is None or maskeddir is None or resultdir is None:
		cwd = getcwd()
		imgdir = cwd + "/data/leedsbutterfly/images"
		maskeddir = cwd + "/processing/masked"
		resultdir = cwd + "/featuretables"
	
	if not exists(resultdir):
		makedirs(resultdir)
	
	resultfile = resultdir + "/bovw.csv"

	# defining feature extractor that we want to use
	extractor = cv2.xfeatures2d.SIFT_create()

	def features(image, extractor):
		keypoints, descriptors = extractor.detectAndCompute(image, None)
		return keypoints, descriptors

	def build_histogram(descriptor_list, cluster_alg):
		histogram = np.zeros(len(cluster_alg.cluster_centers_))
		cluster_result =  cluster_alg.predict(descriptor_list)
		for i in cluster_result:
			histogram[i] += 1.0
		return histogram

	butterflyfiles = ([f for f in listdir(imgdir) if 'png' in f])
	butterflyindices = [f[:-4] for f in butterflyfiles]

	preprocessed_image = []

	#get features and descriptors
	descriptor_list = list()
	i=0
	for butterflyindex in butterflyindices:
		i += 1
		print(f"Image {i}")
		image = cv2.imread(f"{maskeddir}/{butterflyindex}.png", cv2.IMREAD_GRAYSCALE)
		keypoint, descriptor = features(image, extractor)
		#print(descriptor.shape)
		for d in descriptor:
			descriptor_list.append(d)

	print("Performing clustering...")
	kmeans = KMeans(n_clusters = 400)
	kmeans.fit(descriptor_list)

	i=0
	for butterflyindex in butterflyindices:
		i += 1
		print(f"Image {i}")
		image = cv2.imread(f"{maskeddir}/{butterflyindex}.png", cv2.IMREAD_GRAYSCALE)
		keypoint, descriptor = features(image, extractor)
		if (descriptor is not None):
			histogram = build_histogram(descriptor, kmeans)
			preprocessed_image.append(histogram)

	clustercount = range(len(preprocessed_image[0]))
	wordlabels = [f"Word Cluster Number {i}" for i in clustercount]
	wordcounts = np.array(preprocessed_image)
	df = pd.DataFrame(wordcounts, columns = wordlabels)
	df.insert(0, "Butterfly File", butterflyfiles)
	df.to_csv(resultfile, index=False)

if __name__ == '__main__':
	createbovwtable()