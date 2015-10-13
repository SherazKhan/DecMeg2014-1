% Varargin(1)- 'train' or 'test'. Bboth not supported.
% Varargin(2)- List of subjects for training/test
% Varargin(3)- cv enabled or not 'enable'
% Varargin(4)- cv_factor, 0 to 1.

%Quick test code, run this on matlab command line: fd_features_func('train',1,'enable',0.7)

%Parameters
%ml_lib : 'sklearn' will write the preprocessed output as matfile. 'vw'
%will create a VW format. Note VW files are large due to extra class info.

%sensors: list of sensors whoes data is used;
%sfreq: Signal frequency in Hz;
%tmin,tmax: begining and end of time domain signal in sec.
%nSegments: Number of segments if segmentation is used.

function fd_features_func(varargin)

run_mode = varargin(1); %'train','test','both'
subjects_train = varargin(2); 
subjects_test = varargin(2); 
x_val_mode = varargin(3); 

if strcmp(run_mode,'train')
    dataDirName = 'traindata';
    inputFilePrefix='train_subject';
    outputFilePrefix='train_vw_';
    subjects = subjects_train;
else 
    if(strcmp(x_val_mode,'enable'))
        dataDirName = 'traindata';
        inputFilePrefix='train_subject';
        outputFilePrefix='test_vw_';
        subjects = subjects_test;
    else
        dataDirName = 'testdata';
        inputFilePrefix='test_subject';
        outputFilePrefix='test_vw_';
        subjects = subjects_test;
    end
end

ml_lib = 'sklearn';
if (isunix)
        outputFileNamePrefix = strcat('/nfs/wbg_users/kpasad/temp2/',ml_lib,'_data/',outputFilePrefix);
        basepath = strcat('/nfs/wbg_users/kpasad/temp2/',dataDirName,'/');  % Specify absolute path
elseif (ispc)
        outputFileNamePrefix = strcat('C:\Users\kpasad\data\ML\projects\MEGBrainActivity\traindata\',outputFilePrefix);
        basepath = strcat('C:\Users\kpasad\data\ML\projects\MEGBrainActivity\',dataDirName,'\');  % Specify absolute path
else disp('Could not identify OS OS')
end


sensors = 1:306;

sfreq =250;
tmin = 0.5;
tmax = 1;
nSegments = 1;
t_begin = tmin*sfreq+1;
t_end =tmax*sfreq;

% cv_factor = 1 => No x_val
% 0<cv_factor<1 => cv_factor percentage data for training and 1-cv_factor
% percentage for test
% cv_factor =0  => All training data used as test. Used in conjuction with
% test subjects picked from train subjects.

if (nargin ==4)
    cv_factor=varargin{4};
else 
    cv_factor=1;
end
SSS_cut_off= 0.9; % Spectral mask in percentage of single sided spectrum. 1 = full single side spectrum


X=[];
y=[];
train_data =[];
label=[];
test_data =[];
x_val_train_label=[];
x_val_test_idx=1;

%lowpass filter design
Fc    = SSS_cut_off;
N = 10;   % FIR filter order
Hf = fdesign.lowpass('N,Fc',N,Fc);
Hd1 = design(Hf,'window','window',@hamming,'SystemObject',true);
lpf=Hd1.Numerator;

%Notch filter design
f_notch = 50;
Wo = f_notch/sfreq*2;  BW = Wo/50;
%[b,a] = iirnotch(Wo,BW);
    

% Crating the trainset. (Please specify the absolute path for the train data)
disp('Creating the trainset.');
for i = 1 : length(subjects)    
    filename = strcat(basepath,inputFilePrefix,num2str(subjects{i},'%02d'));
    disp(strcat('Loading ',subjects{i}));
    outputFileName = strcat(outputFileNamePrefix,num2str(subjects{i}),'.vw');
    fclose 'all';
    data = load(filename);    
    fileID = fopen(outputFileName,'at');
    XX = data.X; %XX is 594 tests per subject x 306 sensors * length 375 for 1.5 ms data
    if (strcmp(run_mode,'train') || (strcmp(run_mode,'test')&&strcmp(x_val_mode,'enable')))
        yy = data.y; %yy is 594 lables
    end
    clear data;
    
    x_truncated = XX(:,:,t_begin:t_end);
    dims = size(x_truncated);
    n_trials = dims(1);
    n_sensors = dims(2);
    n_samples = dims(3)
    x = reshape(x_truncated,n_trials,n_sensors*n_samples);    
    clear x_truncated
    mean_x= repmat(mean(x),n_trials,1);
    x = x-mean_x;
    clear mean_x;
    std_x = repmat(std(x),dims(1),1);
    x= x./std_x;
    clear std_x;
    disp(strcat('Processing training set ',num2str(i)));
    
        if(strcmp(x_val_mode,'enable')&& strcmp(run_mode,'train'))
            trails_idx = 1:floor(n_trials*cv_factor)  ;
        elseif(strcmp(x_val_mode,'enable')&& strcmp(run_mode,'test'))
            trails_idx = floor(n_trials*cv_factor)+1:n_trials;
        else
            trails_idx = 1:n_trials;  
        end
    
    for trainSampleIdx = trails_idx  
        %disp(strcat('Processing training sample ',num2str(trainSampleIdx)));
        fd_sensor_data=[]; 
        for sensorId=1:length(sensors)                 
         %   sensorData = squeeze(XX(trainSampleIdx,sensorId,:));
         % To turn off segmentation, simply make nSegments as 1
            norm_truncated_sensor_data = x(trainSampleIdx,(1:dims(3))+(sensorId-1)*dims(3))';                                   
            for segment_idx = 1:nSegments
                segment_length = floor(length(norm_truncated_sensor_data)/nSegments);
                segment_begin =(segment_idx-1)*segment_length+1;
                segment_end=segment_begin+segment_length;
                if (segment_end > length(norm_truncated_sensor_data))
                    segment_end = length(norm_truncated_sensor_data);
                end
                segment_data = norm_truncated_sensor_data(segment_begin:segment_end);
                %Notch
                %x_notchFilt = filter(b,a,norm_truncated_sensor_data);                        

                %LPF
                %pwrSD = filter(lpf,1,norm_truncated_sensor_data);             
                %fd_samples = pwrSD(1:4:end);
                %fd_sensor_data=[fd_sensor_data;fd_samples];


                %Pass thru for testing and just adding the labels
                %fd_sensor_data=[fd_sensor_data;norm_truncated_sensor_data];

                %pwrSD = fft(xcorr(norm_truncated_sensor_data));
                %len_SSS = round(SSS_cut_off*length(pwrSD)/2);
                %DSS_window_idx = round(SSS_cut_off*length(pwrSD)/2):(length(pwrSD)-round(SSS_cut_off*length(pwrSD)/2));
                %DSS_window = ones(1,length(pwrSD));
                %DSS_window(DSS_window_idx)=0;

                %Frequency domain windowing and no TD conversion. Only one side band is used
                %pwrSD = fft(segment_data);
                %len_SSS = round(SSS_cut_off*length(pwrSD)/2);
                %fd_samples = real(pwrSD(1:len_SSS));
                %fd_sensor_data=[fd_sensor_data;fd_samples];

                %Frequency domain windowing and no TD conversion. Two side band is used
                pwrSD = fft(segment_data);
                fd_samples = real(pwrSD);
                fd_sensor_data=[fd_sensor_data;fd_samples];
                
                %Frequency Domian windowing and time domain conversion
%                 pwrSD = fft(segment_data);
%                 DSS_window_idx = round(SSS_cut_off*length(pwrSD)/2):(length(pwrSD)-round(SSS_cut_off*length(pwrSD)/2));
%                 DSS_window = ones(1,length(pwrSD));
%                 DSS_window(DSS_window_idx)=0;                                           
%                 fd_samples = real(fft(pwrSD.*DSS_window'))/sqrt(length(pwrSD));
%                 fd_sensor_data=[fd_sensor_data;fd_samples(1:4:end)];
                
            end %loop over segments
        end  %Loop over sensors    
        
        % If training, then capture the training labels. Only CV % of
        % training labels are accumalated due to CV %trainingSample loop
        if strcmp(run_mode,'train')
            lable_str = [num2str(2*yy(trainSampleIdx)-1),' |','feat'];
            label  = [label;(2*yy(trainSampleIdx)-1)];
        else %If CV is used, then 1-CV train samples label need to be stored. If CV is not enabled, the labels are dummy 
            lable_str = [num2str(1),' |','feat'];%dummy 1
            label  = [label;1];
            if (strcmp(x_val_mode,'enable'))
                x_val_train_label(x_val_test_idx)=yy(trainSampleIdx);
                label(x_val_test_idx)=yy(trainSampleIdx);
                x_val_test_idx=x_val_test_idx+1; %Universal counter
            end
        end
        if (strcmp(ml_lib,'sklearn'))
            train_data = [train_data;fd_sensor_data'];        %Matlab stacked array, row = train sample, col = feature
        else
            strVal= num2str(fd_sensor_data,'%2.5e');       
            strFeatIdx = num2str([0:length(fd_sensor_data)-1]','%05d');
            strSep = repmat(':',length(fd_sensor_data),1);
            strSpace=blanks(length(fd_sensor_data))';
            str_vw = [strSpace,strFeatIdx,strSep,strVal];         
            c=cellstr(str_vw);
            fprintf(fileID,lable_str);
            fprintf(fileID,cell2mat(c'));
            fprintf(fileID,'\n');                
        end
        
       end %Loop over train_data                      
    
end
    
    if(strcmp(x_val_mode,'enable')&& strcmp(run_mode,'test')&&strcmp(ml_lib,'vw'))
        save(strcat('vw_data/','x_val_train_label_',num2str(subjects{i}),'.mat'),'x_val_train_label');
    end
    if (strcmp(ml_lib,'sklearn')&&strcmp(run_mode,'train'))
         save(strcat('sklearn_data/','train_sklearn_',num2str(subjects{i}),'.mat'),'train_data','label');       
    elseif (strcmp(ml_lib,'sklearn')&&strcmp(run_mode,'test'))
        save(strcat('sklearn_data/','test_sklearn_',num2str(subjects{i}),'.mat'),'train_data','label');       
    end
    fclose 'all'
end
