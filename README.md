# DecMeg2014
DecMeg2014 - Decoding the Human Brain Kaggle competition
The project details can be found here:
https://www.kaggle.com/c/decoding-the-human-brain

The code provided here is a straight forward application of stacked generalisation ensembling as mentioned in the project. The code is organised as follows:

Run the matlab script xxx.m. xxx.m calls the preprocessing function in file yyy.m. The outut is train.mat and test.mat

Run xxx.py to generate per subject models.

Run the cross validation

Run xxx. to create a submission.

What does the code do?

Stacked genralisation is a method of ensembling where predictions from multiple predictors are ensembled using another predictor.

Each subject, Si, i=0-15 is trained on data <xi,yi>, and xi = {xi,o...xi,mi} resulting in a model mi.
