A new version of the Butterfly Classification project to make it easier for other people to run.

To process all the files and tests:
First, install all the necessary dependencies (OpenCV2, numpy, sklearn, mlxtend, pandas, ripser, and persim).
Note that in OpenCV2, the SIFT algorithm is patented, and so is not available through package installers. You will have to compile OpenCV2 with the nonfree algorithms manually (see https://stackoverflow.com/questions/50467696/pycharm-installation-of-non-free-opencv-modules-for-operations-like-sift-surf)
For the rest of the dependencies, I recommend using pip with a venv.

Second, run python src/processing/main.py. This will run all the of the methods needed to build up to the persistence images automatically (and will store the intermediate files in case you need to use them or make changes to only parts of the process later)

Third, run python src/tablecreation/main.py This will create the CSV tables for the bag-of-visual-words feataures and the persistence image features.

Finally, run python src/algorithms/main.py This will prepare the learning models' hyperparameters for each feature training set (persims, bovw, and both) and run tests to determine their efficacy. It will also compare the accuracies of the classifiers when using bovw and when using both, and check for statistical significance using a slightly-modified version of mlxtend's http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/.
