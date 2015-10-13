# This file generates the training models for ever subject. i.e. it regresses the subject specific training data on subject specific outcome. 

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import subprocess
    
def write_vw(x,y,mode,outfile):
    lines_wrote = 0
    for trial_nr, X in enumerate(x):
        outline = ""
        if mode=='train':
            if y[trial_nr] == 1:
                label = 1
            else:
                label = -1 #change label from 0 to -1 for binary
            outline += str(y[trial_nr]) + " '" + str(lines_wrote) + " |f"
        else:
            label = 1 #dummy label for test set       
            outline += str(label) + " '" + " |f"

        for feature_nr, val in enumerate(X):
            outline += " " + str(feature_nr) + ":" + str(val)
        outfile.write( outline + "\n" )
        lines_wrote += 1
        if trial_nr % 100 == 0:
            print "%s\t%s" % (trial_nr,  lines_wrote)
    

train_file = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train.mat" 
test_file  = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test.mat"


#Initialise variables
subject_err = np.empty([16,16])  

train_data = loadmat(train_file, squeeze_me=True)
test_data =loadmat(test_file, squeeze_me=True)
n_subjects = np.unique(train_data['subject_id']).size
dims = train_data['label'].shape
n_train = dims[0]

x_train = train_data['train_data']
y_train = train_data['label']

vw_exe = 'C:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw.exe'
trainCommandParams = ('-c -k --passes 60 --loss_function logistic' )
testCommandParams = ('-t' )

#Create split vw files for each subject
for train_subject_id in range(1,n_subjects+1):    
    x_train = train_data['train_data'][np.where(train_data['subject_id']==train_subject_id)]
    y_train = train_data['label'][np.where(train_data['subject_id']==train_subject_id)]
    print "Creating l0 train data per subject",train_subject_id    
    train_l0_data = open('c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_l0_data_vw_'+str(train_subject_id)+'.txt','w')        
    write_vw(x_train,y_train,'train',train_l0_data)


#train Models on each subject
for train_subject_id in range(1,n_subjects+1):       
    train_l0_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_l0_data_vw_'+str(train_subject_id)+'.txt'        
    l0_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/l0_model_vw_'+str(train_subject_id)+'.vw'        
    l0_train_command = []    
    l0_train_command = [vw_exe,'-d', train_l0_data, '-f',l0_model]
    l0_train_command.extend(trainCommandParams.split(' '))
    
    print "Training L0 model for subject ",train_subject_id
    with open('myfile', "w") as stdoutfile:
        subprocess.call(l0_train_command, stdout=stdoutfile)

