# ML Project 1 

### A New Regularized Logistic Regression Method to detect Higgs Boson production
This project aims to find the best classification model to detect Higgs Boson decay signatures in a dataset of collisition decay signatures.

Files Description:

1. ```implementations.py``` This file collects different implementations of machine learning techniques, such as least squares, linear gradient descent, linear stochastic gradient descent, ridge regression, logistic regression and regularized logistic regression. 
2. ```helpers.py``` This file gathers different functions to help sample the data used to train our model. 
3. ```utils.py``` Diverse utility functions.
4. ```cross_validation.py``` Cross-validation functions. Cross-validation is used to compute the loss and the accuracy of model prediction on unknown data. Different variants of cross-validation are implemented, including functions to find the best value of a particular parameter for a given training technique. 
5. ```run.py``` Python script to generate the ```submission.csv``` of the predictions made on the test set, ```test.csv```.
6. ```Project1_new.ipynb``` Notebook to keep trace of our research, includes among others different runs of cross_validations, the methods to visualize the train dataset. This is interesting to look if one is interested in the way the data figures on the report are made.
7. ```data/train.csv``` Train dataset with known classifications. This file is initially not present, please put it there.
8. ```data/test.csv``` Test dataset with unknown classifications.
8. ```data/sample-submission.csv``` Example of a submission file.


How to: 

Simply run the ```run.py``` in the cloned repository, making sure that the ```data``` folder sits in the folder of the ```run.py```. This will create a ```submission.csv``` in the ```data``` folder.
Note that it takes a few minutes to generate the file (2-3 minutes)
