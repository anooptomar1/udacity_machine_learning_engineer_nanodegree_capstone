# Udacity Machine Learning Engineer Nanodegree Capstone Project

## Structure 

### Report 
- Final_Report.pdf 

### Implementation 
- subset_labels.csv; subset of the data, used for training and testing 
- test_set.json; defines the test set 
- sketch_recognition_trainer.py; defines the bulk of the application from feature extraction to training
- misc_utils.py; utility methods 
- feature_extractors; Collection of feature extractors 
- classifiers; classifier wrappers for sketch classification
- tuning; model tuning classes 
- evaluators; encapsulates evaluating the features and model
  
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
