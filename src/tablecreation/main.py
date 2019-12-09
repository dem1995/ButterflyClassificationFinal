from os import getcwd
from time import sleep

from classifications import createclassificationtable
from bovw import createbovwtable
from persims import createpersimtable

def maketables():
	cwd = getcwd()
	imgdir = cwd + "/data/leedsbutterfly/images"
	maskeddir = cwd + "/processing/maskedcropped"
	resultdir = cwd + "/featuretables"
	persimdir = cwd + "/processing/persims"

	print("Creating classification table")
	createclassificationtable(imgdir=imgdir, resultdir=resultdir)

	print("Creating bag-of-visual-words feature table")
	createbovwtable(imgdir=imgdir, maskeddir=maskeddir, resultdir=resultdir)

	print("Creating persistence image feature table")
	createpersimtable(imgdir=imgdir, persimdir=persimdir, resultdir=resultdir)

if __name__ == '__main__':
	maketables()