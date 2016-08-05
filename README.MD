# Udacity Machine Learning Engineer Nanodegree Capstone Project

## Structure 

### Report 
- Final_Report.pdf

### Working prototype
https://sketching.herokuapp.com/sccanvas.html

### Running the project 
1. Download the dataset from http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
2. Unarchive the contents into the root folder where this project resides i.e. the extracted folder 'png' should be at the same level as this project (sketchrecognition)
3. Change directory into the project directory
4. Run 'python rebuild_models.py' to rebuild models and run the run the evaluation (which will reside in the results directory when complete)

### Implementation 
- subset_labels.csv; subset of the data, used for training and testing 
- test_set.json; defines the test set

- constants.py; set of global constants  
- misc_utils.py; utility methods for IO and image loading and manipulation
- sketch_recognition_trainer.py; wrapper for building the codebook and training the classifier 
- feature_extractors.py; responsible for extracting the descriptors from a image 
- sketch_classifier.py; set of classifier wrappers for classification
- tuning; feature and model tuning classes
-- tuning_helper.py; helper methods for tuning 
-- feature_engineering_tuning.py; set of 'explorationary experiments' to observe the influence each feature property have on the performance
-- model_tuning.py; similar to above but focusing on the performance of the classifier model
-- replay.py; used to rebuild and confirm test results published in report  
- evaluator.py; encapsulates evaluating the features and model
  
- results; directory where evaluation results are written to 
- data; output from the feature extraction and model building steps (including codebook) 

### Report Material (iPython Notebooks) 
- 1 - Problem Definition.ipynb
- 2 - Exploratory Analysis (feature extraction and engineering).ipynb
- 3 - Training (Model selection and tuning) and Evaluation.ipynb

## References
Work based on http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

## Dataset 
http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip

## Depedencies 
- matplotlib 1.5
- numpy 1.11.1
- scikit-learn 0.17.1
- scipy 0.17.1
- OpenCV 2.4.13

### Installing OpenCV (Mac) 
Python home: /Users/josh/.virtualenvs/venv_sketchrec/bin/python

$ brew install opencv --with-tbb --with-opengl
$ cd ~/.virtualenvs/venv_sketchrec/lib/python2.7/site-packages/
$ ln -s /usr/local/Cellar/opencv/2.4.13/lib/python2.7/site-packages/cv.py
$ ln -s /usr/local/Cellar/opencv/2.4.13/lib/python2.7/site-packages/cv2.so  
