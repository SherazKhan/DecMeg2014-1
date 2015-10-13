# DecMeg2014
DecMeg2014 - Decoding the Human Brain Kaggle competition
The project details can be found here:
https://www.kaggle.com/c/decoding-the-human-brain

The code provided here is a straight forward application of stacked generalisation ensembling as mentioned in the project. The code is organised as follows:

1. Run the matlab script fd_features_func.m. This scripts will generate the train.mat and test.mat, which are preprocessed matlab files. Preprocessing includes (not all options are enabled, some may be commented off): 
    - Notch filtering to remove 60Hz powerline interference
    - Low pass filtering
    - Windowing
    - Downsampling
    - FFT to generate coefficients frequency domain coefficients

2. Run xxx.py. xxx.py runs a logistic regression individually for every subject. Each subject, Si, i=0..15 is trained on data <xi,yi>, where xi is the matrix of feaures and yi is vector of labels. The number of training data for i^th subject is Ni. Each subject has a sligthly diffrent number of trainig data. For each subject, the resulting model is mi. 

3.Run yyy.py. The entire training set <x,y> is regressed, over all models mi, i= 0 to 15. The result is  a matrix of 16 columns (one for each mi) and Ntrain rows. These are the level 0 predictions, also called level 1 training data.

4. yyy.py uses the level 0 predictions from step 3 as training data, along with the original lables , to create a stacked generalisation emsemble. I use logistic regression as ensemble as well.
5. The learning algorithms are executed via Vowpal_wabbit. By changing the loss functions a diffrent learning model can be applied for both, step 2/3 and 4 above.
6. yyy.py is used for cross validation- a leave one out CV. One subject is the used as test. We iterate through all users as test candidates.
7. Run xxx. to create a submission- xxx.py does all of the above steps but instead of CV, it generates outcomes on the test data
8. As is, this code get you into top 25%. There is plenty of potential oppurtunities for improvements. Some low hanging fruits include regularisation, diffrent base and ensembling algorithms. 
9. The preprocessing is suboptimal- FFT is suboptimal for stochastic signals. A eigen decomposition and eigen filtering should yeild a more generalised feature set.
