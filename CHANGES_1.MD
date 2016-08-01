# Changes 30/07/2016 

## Definition
**Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.**   
*Please provide the mathematical definition of the logloss function so that it is easier to interpret your results in later sections.*  

Added Log Loss formula. 

## Analysis  
**A visualization has been provided that summarizes or extracts a relevant characteristic or feature about the dataset or input data with thorough discussion. Visual cues are clearly defined.**  
*A minor correction here:  
Most of the bar charts in the feature engineering section seem to have the wrong captions (Accuracy, Cluster size, K Accuracy %).*    

Amending the captions for each of the plots (Accuracy, Log Loss, Accuracy for the top predictions)  

**Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.**  
*It is not entirely clear how the bag of visual words workflow works. The workflow figure in the Feature Extraction section is a good start but you need to provide more details. Some leading questions:  
- How big is the dimensionality of the space where you perform clustering? Is that the 128 bit feature space?  
- After you perform clustering, how do you build histograms for each image?*  
  
Re-wrote the section describing the process, going into more details and including a highlevel overview of the process (and how it integrates into training and classification).   

## Results 
**The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.**
*Since you use a subset of 113 categories for developing your method, you are left with a large chunk of the data unused. This is a great opportunity to test the robustness of your classifier. You can pick the best model you have developed so far and fit it on a new subset of the categories and see if its classification performance remains the same.*  

Retrained and tested against 113 categories from the leave-out data, evaluated and compared with the initial training set. 