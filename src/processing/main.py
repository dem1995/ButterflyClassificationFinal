from os import getcwd
from os.path import exists
from time import sleep

from imgmasking import maskimages
from imgresizing import resizeimages
from imgfiltering import filterimages
from persdiagcreate import createpersdgmsfrombinarizedimages
from persimcreate import createpersimsfrompersdgms


def preprocess_images_to_get_persims():

	cwd = getcwd()
	imgdir = cwd + "/data/leedsbutterfly/images"
	maskdir = cwd + "/data/leedsbutterfly/segmentations"
	maskedcroppedimgdir = cwd + "/processing/maskedcropped"
	resizedimgdir = cwd + "/processing/resized"
	filteredimgdir = cwd + "/processing/filtered"
	persdgmdir = cwd + "/processing/persdgms"
	persimdir = cwd + "/processing/persims"

	delay = 1.1

	#Mask and crop images
	print("Masking and cropping images...")
	sleep(delay)
	if not exists(maskedcroppedimgdir):
		maskimages(imgdir, maskdir, maskedcroppedimgdir)
	else:
		print("Images were already masked and cropped.")
	
	sleep(delay)

	#Resize images
	print("Resizing images...")
	sleep(delay)
	if not exists(resizedimgdir):
		resizeimages(maskedcroppedimgdir, resizedimgdir)
	else:
		print("Images already resized")

	sleep(delay)

	#Filter images
	print("Filtering images...")
	sleep(delay)
	if not exists(filteredimgdir):
		filterimages(resizedimgdir, filteredimgdir)
	else:
		print("Images already filtered.")

	sleep(delay)

	#Calculate persistence diagrams
	print("Creating persistence diagrams.")
	sleep(delay)
	createpersdgmsfrombinarizedimages(filteredimgdir, persdgmdir)

	sleep(delay)

	#Calculate persistence images
	print("Creating persistence images.")
	sleep(delay)
	createpersimsfrompersdgms(persdgmdir, persimdir)

if __name__ == '__main__':
	preprocess_images_to_get_persims()