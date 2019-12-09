'''
Masks and crops the butterfly images using their segmentation masks.
'''

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname
import shutil
import tempfile

import cv2
import numpy as np

def maskimages(imgdir, maskdir, resultdir):
	'''
	imgdir is the absolute filepath of the directory to get images from.
	maskdir is the absolute filepath to get the segmentation masks from
	WARNING: see below
	resultdir is the absolute filepath to which to store the processed images.
	WARNING: resultdir (the lowest level of the filepath, that is, not all of your system obviously) will be wiped if it already exists
	Default subdirectories of cwd (which itself defaults to using getcwd()) are used if these are None.
	'''

	#make a directory to store the masked images in
	if exists(resultdir):
		tmp = tempfile.mktemp(dir=dirname(resultdir))
		# Rename the dir.
		shutil.move(resultdir, tmp)
		# And delete it.
		shutil.rmtree(tmp)
	makedirs(resultdir)


	imagefiles = [f for f in listdir(imgdir)]

	#mask the images and put them in the directory
	for imagefile in imagefiles:
		#Get the image and the mask	
		image = cv2.imread(imgdir+"/"+imagefile)
		image = cv2.bitwise_not(image)

		mask = cv2.imread(maskdir+"/"+imagefile[:-4]+"_seg0.png", 0)
		#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		(thresh, mask) = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
		
		#Apply the mask
		maskedimage = cv2.bitwise_and(image, image, mask=mask)
		maskedimage = cv2.bitwise_not(maskedimage)

		#Get the x and y coordinates of the mask start/stop for cropping
		#i.e., determine the cropping boundaries
		minx = mask.shape[1]
		miny = mask.shape[0]
		maxx = 0
		maxy = 0
		for row in range(mask.shape[0]):
			for column in range(mask.shape[1]):
				if mask[row][column]==255:
					if column < minx: minx = column
					if column > maxx: maxx = column
					if row < miny: miny = row
					if row > miny: maxy = row

		#Shift the crop boundaries outward a bit to make sure the to-be-applied Sobel (or other) filters
		#don't miss the extreme parts of the butterflies
		marg = 6
		minx = minx-marg if minx>=0+marg else 0
		miny = miny-marg if miny>=0+marg else 0
		maxx = maxx+marg if maxx<mask.shape[1]-marg else mask.shape[1]-1
		maxy = maxy+marg if maxy<mask.shape[0]-marg else mask.shape[0]-1

		#Crop the image and output it to a file
		maskedimage = maskedimage[miny:maxy, minx:maxx]
		cv2.imwrite(resultdir+"/"+imagefile, maskedimage)
