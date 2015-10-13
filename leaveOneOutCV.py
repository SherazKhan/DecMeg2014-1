#This file performs a Leave one out cross validation. One out of the 16 subjects is left out as a test data. 
# We loop over all subjects: each subject is a candidate for LOO subject. At the end we have the cross validation for every subject.
# PREREQUSITE: To run this script, er subject model mus be available in VW format. They can be generated by gen_per_subject_model.py.
# Alternatively, a model generated by other means should be available in VW format.


import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import subprocess
import pickle

def generate_submission(loc_preds, loc_submission, header="Prediction", binary=True):
    with open(loc_submission, "wb") as outfile:
        if len(header) > 0:
            outfile.write(header+"\n")

        for e, line in enumerate( open(loc_preds) ):
            row = line.strip().split(" ")
            if binary:
                if float(row[0]) >= 0:
                       #if float(row[0]) == 0:
                    pred = "1"
                else:
                    pred = "0"
            #outfile.write(row[1]+","+pred+"\n")
            outfile.write(pred+"\n")

    
    
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
    


#Location of preprocessed training and test data. Pre processing is in MATLAB.
train_file = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/train.mat" 
test_file  = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/test.mat"


vw_exe = 'C:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw.exe'
trainCommandParams = ('-c -k --passes 60 --loss_function logistic' )
testCommandParams = ('-t' )

#Initialise variables
subject_err = np.empty([16,16])  

#Load the data
print 'loading train...'
train_data = loadmat(train_file, squeeze_me=True)
print 'loading test'
test_data =loadmat(test_file, squeeze_me=True)


n_subjects = np.unique(train_data['subject_id']).size
dims = train_data['label'].shape
n_train = dims[0]

x_train = train_data['train_data']
y_train = train_data['label']

list_of_subject = range(1,n_subjects+1)
subject_error=np.empty([n_subjects])

for loocv_id in range(1,n_subjects+1):
    list_of_subject = range(1,n_subjects+1)
    loocv_subject = list_of_subject.pop(loocv_id-1) #List of subjects has one less subject (loocv_subject)
    
    print '---Cross Validation : Leaving out subject ',loocv_subject
    
   
    #Extract subject train data. This is the entire train set SANS the left out subject...
    x_train = train_data['train_data'][np.where(train_data['subject_id']!=loocv_subject)]
    y_train = train_data['label'][np.where(train_data['subject_id']!=loocv_subject)]
    weight  =cov_shift[np.where(train_data['subject_id']!=loocv_subject)]
    
	#...and write it in VW format
	train_l0_data_vw = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/train_vw_cs_'+str(loocv_subject)+'.vw'        
    write_vw(x_train,y_train,'train',open(train_l0_data_vw,'w'))
    n_train = len(x_train)

    #Extract the CV candidate test subject
    x_test = train_data['train_data'][np.where(train_data['subject_id']==loocv_subject)]
    y_test = train_data['label'][np.where(train_data['subject_id']==loocv_subject)]
    
	#...and write it in VW format
    test_l0_data_filename = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/test_l0_data_vw_'+str(loocv_subject)+'.txt'
    test_l0_data = open(test_l0_data_filename,'w')        
    write_vw(x_test,y_test,'test',open(test_l0_data_vw,'w'))
    
    
    l1_train_data = np.empty([n_train,n_subjects])    
    
    # A model for every subject is precomputed. Run the reduced train set per subject models. 
    for subject_id in list_of_subject:      #Note: The CV candidate subject has been removed previously
        l0_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/l0_model_vw_'+str(subject_id)+'.vw'   #pre-computed model for each subject
        train_l1_data_per_subject = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/train_l1_vw_temp.txt'   #File to hold VW Classifier output      
        l0_test_command = []    
        l0_test_command = [vw_exe, '-d', train_l0_data_vw ,'-i', l0_model , '-p',train_l1_data_per_subject, '-t' ] #Apply the model and save the result in train_l1_data file
        print "Generating l1 meta-data component from subject %d",subject_id        
        with open('myfile', "w") as stdoutfile:
            subprocess.call(l0_test_command, stdout=stdoutfile) #run the classifier
        
        print "Converting into probalilities.."

        with open(train_l1_data_per_subject) as f:
            content = f.readlines()
            for n_train in range(0,len(content)-2): 
                prob_string = content[n_train].strip().split(" ")
                l1_train_data[n_train,subject_id-1]=1/(1+np.exp(-float(prob_string[0])) ) #One feature (the left out subject) is zero always. Its OK.      
    
    #At this point all L0 soft decisions for all except one subjects are available. 
    train_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/train_l1_metadata_vw.txt'  
    write_vw(l1_train_data,y_train,'train',open(train_l1_data,'w'))
    
    #Create l1 Meta Model for all except one     
    
    l1_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/l1_meta_model_vw.vw' #Stacked generalisation model to be created
    l1_command_params = ('-c -k --passes 60 --loss_function logistic' )
    l1_train_command = []  
    l1_train_command = [vw_exe,'-d', train_l1_data, '-f', l1_model] #l0 classifier prob. values form the L1 train data
    trainCommandParams = ('-c -k --passes 60 --loss_function logistic' )
    l1_train_command.extend(trainCommandParams.split(' '))
    with open('myfile', "w") as stdoutfile:
        subprocess.call(l1_train_command, stdout=stdoutfile)
       
    #_____________Training complete_______________________
	
    #Put the test throught the same processing:
    dims = x_test.shape
    n_test = dims[0]
    l1_test_data = np.empty([n_test,n_subjects])   
    test_l1_data_per_subject = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/test_l1_metadata_vw.txt'  #VW file for L1 data for CV subject                 
    
    test_l0_data_vw = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/test_vw_'+str(loocv_subject)+'.vw'  # Test data for CV subject  

    for subject_id in list_of_subject:            
        l0_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/l0_model_vw_'+str(subject_id)+'.vw'        
        l0_test_command = []    
        l0_test_command = [vw_exe, '-d', test_l0_data_vw ,'-i', l0_model , '-p',test_l1_data_per_subject, '-t' ] # Regress CV subject 
        print "Creating L1 meta data for test from subject  ",subject_id    
        with open('myfile', "w") as stdoutfile:
            subprocess.call(l0_test_command, stdout=stdoutfile)    
    
        with open(test_l1_data_per_subject) as f:
            content = f.readlines()
            for n_train in range(0,len(content)-2): 
                prob_string = content[n_train].strip().split(" ")
                l1_test_data[n_train,subject_id-1]=1/(1+np.exp(-float(prob_string[0]) ))       
    
	test_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/test_l1_metadata_vw_full.txt'
    write_vw(l1_test_data,y_test,'test',open(test_l1_data,'w'))
	#At this point the test subject has been regressed on all individual l0 models and l1 data for test subject is ready				
    #Run l1 data through l1 model                
    print "Running test L1 meta data throuth L1 meta model"
    l1_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full_loocv/l1_meta_model_vw.vw'
    l1_test_command = []    
    l1_test_command = [vw_exe, '-d', test_l1_data ,'-i', l1_model , '-p','c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/face.predict.txt', '-t' ]
    
    with open('myfile', "w") as stdoutfile:
        subprocess.call(l1_test_command, stdout=stdoutfile)

    error=np.empty([n_test])
    for e, line in enumerate( open('c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/face.predict.txt') ):
            row = line.strip().split(" ")
            if float(row[0]) >= 0:
                pred = 1
            else:
                pred = 0
                
            error[e] = abs(pred-(y_test[e]+1)/2)    
    subject_error[loocv_id-1]=error.mean()
    print subject_error