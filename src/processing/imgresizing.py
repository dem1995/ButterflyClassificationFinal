'''
Resizes images to make sure they don't have a computationally onerous number of points for processing.
'''

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname
import shutil
import tempfile

import cv2
import numpy as np


def resizeimages(imgdir, resultdir):
	'''
	imgdir is the absolute filepath of the directory to get images from.
	WARNING: see below
	resultdir is the absolute filepath to which to store the processed images.
	WARNING: resultdir (the lowest level of the filepath, that is, not all of your system obviously) will be wiped if it already exists
	'''

	#make a directory to store the resized images in
	if exists(resultdir):
		tmp = tempfile.mktemp(dir=dirname(resultdir))
		# Rename the dir.
		shutil.move(resultdir, tmp)
		# And delete it.
		shutil.rmtree(tmp)
	makedirs(resultdir)

	imagefiles = [f for f in listdir(imgdir)]

	#resize the images and put them in the directory
	for imagefile in imagefiles:
		#Get the image
		image = cv2.imread(imgdir+"/"+imagefile)

		#Determine a suitable scaling factor for the image
		resizedimage=image
		longerdim = resizedimage.shape[0] if resizedimage.shape[0]>resizedimage.shape[1] else resizedimage.shape[1]
		scalefactor = 1
		while longerdim*scalefactor > 100:
			scalefactor = scalefactor * 0.9

		#Scale the image and write it out to a file.
		resizedimage = cv2.resize(image, None, fx=scalefactor, fy=scalefactor, interpolation = cv2.INTER_AREA)
		cv2.imwrite(resultdir+"/"+imagefile, resizedimage)