"""
Python program for filtering color butterfly images to retrieve black-and-white representations of prominent edges.
Uses Sobel filters to accomplish this.
"""

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists, dirname
import shutil
import tempfile

import cv2
import numpy as np

def filterimages(imgdir, resultdir):
	'''
	imgdir is the absolute filepath of the directory to get images from.
	resultdir is the absolute filepath to which to store the processed images.
	Default subdirectories of cwd (which itself defaults to using getcwd()) are used if these are None.
	'''

	#make a directory in which to store the filtered images
	if exists(resultdir):
		tmp = tempfile.mktemp(dir=dirname(resultdir))
		# Rename the dir.
		shutil.move(resultdir, tmp)
		# And delete it.
		shutil.rmtree(tmp)
	makedirs(resultdir)

	imagefiles = [f for f in listdir(imgdir)]

	#Filter the images and put them in the directory
	for imagefile in imagefiles:

		#Get the image, convert it to greyscale, and blur it to remove noise
		image = cv2.imread(imgdir+"/"+imagefile)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(grey,(1,1),0)

		#Apply sobel filters to pick up on prominent edges in the image
		sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)  # x
		sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)  # y
		dx, dy = sobelx, sobely
		mag = np.hypot(dx, dy)  # magnitude
		filteredimage = mag * 255.0 / np.max(mag)  # normalize

		#Write the intermediate greyscale, prominent-edges-picked-up image out to a file
		cv2.imwrite(resultdir+"/"+imagefile, filteredimage)

		#Threshold the greyscale, prominent-edges-picked-up image so that it is black and white
		imgmean = np.mean([pixel for pixel in np.ndarray.flatten(filteredimage) if pixel is not 0])
		print(imgmean)
		(thresh, filteredimage) = cv2.threshold(filteredimage, 20, 255, cv2.THRESH_BINARY)
		filteredimage = 255-filteredimage
		
		#Write the final, prominent-edges-picked-up black-and-white image out to a file
		cv2.imwrite(resultdir+"/"+imagefile[0:-4]+"thresh.png", filteredimage)
