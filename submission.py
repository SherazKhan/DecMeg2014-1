import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import subprocess
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
    

train_file = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train.mat" 
test_file  = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test.mat"
train_l0_data_vw = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_vw.vw'        
test_l0_data_vw = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test_vw.vw'

vw_exe = 'C:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw.exe'
trainCommandParams = ('-c -k --passes 60 --loss_function logistic' )
testCommandParams = ('-t' )

#Initialise variables
subject_err = np.empty([16,16])  

train_data = loadmat(train_file, squeeze_me=True)
test_data =loadmat(test_file, squeeze_me=True)
n_subjects = np.unique(train_data['subject_id']).size
dims = train_data['label'].shape
n_train = dims[0]

x_train = train_data['train_data']
y_train = train_data['label']
l1_train_data = np.empty([n_train,n_subjects])


#Create  vw files to hold ALL train and all test data
x_train = train_data['train_data']
y_train = train_data['label']
print "Creating vw l0 train data "
write_vw(x_train,y_train,'train',open(train_l0_data_vw,'w'))   
print "Creating vw l0 test data "
x_test = test_data['train_data']
write_vw(x_test,y_train,'test',open(test_l0_data_vw,'w'))   

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

# Run the entire train data on per subject models

for subject_id in range(1,n_subjects+1):            
    l0_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/l0_model_vw_'+str(subject_id)+'.vw'        
    train_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_l1_vw_temp.txt'        
    l0_test_command = []    
    l0_test_command = [vw_exe, '-d', train_l0_data_vw ,'-i', l0_model , '-p',train_l1_data, '-t' ]
    print "Generating l1 meta-data component from subject %d",subject_id        
    with open('myfile', "w") as stdoutfile:
        subprocess.call(l0_test_command, stdout=stdoutfile)
    
    print "Converting into probalilities.."

    with open(train_l1_data) as f:
        content = f.readlines()
        for n_train in range(0,len(content)-2): 
            prob_string = content[n_train].strip().split(" ")
            l1_train_data[n_train,subject_id-1]=1/(1+np.exp(-float(prob_string[0])) )       

#Convert L1 meta data to VW file and train another LR  
y_train = train_data['label']
train_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_l1_metadata_vw.txt'                  
write_vw(l1_train_data,y_train,'train',open(train_l1_data,'w')                  )

#Create l1 Meta Model    
train_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/train_l1_metadata_vw.txt'                  

l1_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/l1_meta_model_vw.vw'
l1_command_params = ('-c -k --passes 60 --loss_function logistic' )
l1_train_command = []  
l1_train_command = [vw_exe,'-d', train_l1_data, '-f', l1_model]
trainCommandParams = ('-c -k --passes 60 --loss_function logistic' )
l1_train_command.extend(trainCommandParams.split(' '))
with open('myfile', "w") as stdoutfile:
    subprocess.call(l1_train_command, stdout=stdoutfile)
   

#Put the test throught the same processing:
dims = test_data['label'].shape
n_test = dims[0]
l1_test_data = np.empty([n_test,n_subjects])   
test_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test_l1_metadata_vw.txt'                  

for subject_id in range(1,n_subjects+1):            
    l0_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/l0_model_vw_'+str(subject_id)+'.vw'        
    l0_test_command = []    
    l0_test_command = [vw_exe, '-d', test_l0_data_vw ,'-i', l0_model , '-p',test_l1_data, '-t' ]
    print "Creating L1 meta data for test from subject  ",subject_id    
    with open('myfile', "w") as stdoutfile:
        subprocess.call(l0_test_command, stdout=stdoutfile)    

    with open(test_l1_data) as f:
        content = f.readlines()
        for n_train in range(0,len(content)-2): 
            prob_string = content[n_train].strip().split(" ")
            l1_test_data[n_train,subject_id-1]=1/(1+np.exp(-float(prob_string[0]) ))       
            
test_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test_l1_metadata_vw_full.txt'
write_vw(l1_test_data,y_train,'test',open(test_l1_data,'w'))

print "Running test L1 meta data throuth L1 meta model"
l1_model = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/l1_meta_model_vw.vw'
test_l1_data = 'c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/vw_data_fd_full/test_l1_metadata_vw_full.txt'
l1_test_command = []    
l1_test_command = [vw_exe, '-d', test_l1_data ,'-i', l1_model , '-p','c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/face.predict.txt', '-t' ]

with open('myfile', "w") as stdoutfile:
    subprocess.call(l1_test_command, stdout=stdoutfile)

loc_preds = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/face.predict.txt"
loc_submission = "c:/Users/kpasad/data/ML/projects/MEGBrainActivity/traindata/submissions.csv"
generate_submission(loc_preds, loc_submission)

