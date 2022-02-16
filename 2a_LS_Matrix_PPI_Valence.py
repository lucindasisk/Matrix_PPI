#!/usr/bin/env python
# coding: utf-8

# # Matrix PPI
# ### ABCD Nback data, Shen parcellation atlas

# In[3]:


import pandas as pd
from nipype.interfaces import spm
from glob import glob
from nipype.algorithms.modelgen import spm_hrf
from numpy import nan, convolve, asarray, savetxt, dot, ones, shape
import numpy as np
from numpy.linalg import lstsq, solve
from scipy.stats import zscore, linregress
from datetime import date
from os.path import join
import csv
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
today=str(date.today())


# ### Set directories

# In[4]:


user = 'mrrc'

if user == 'laptop':
    #home='/Users/lucindasisk/Box/LS_Folders/CANDLab/Projects/ABCD_Stress_NBack/DATA'
    home='/Users/lucindasisk/Box/LS_Folders/CANDLab/Projects/ABCD_Stress_NBack/HCP_QC/Sample_Data'
    reg_dir = home #join(home, 'LS_regressors')
    out_dir = home #+ '/newtest' #join(home, 'LS_PPI_connectivity')
    mat_date = '2020-03-04' #Date that matrix regressors were generated
    subs = ['sub-SAMPL000000']

elif user == 'mrrc':
    home='/mnt/abcd/code/setupfiles/LS_ABCD_368_with_GSR_LP1'
    reg_dir = join(home, 'LS_regressors')
    out_dir = join(home, 'LS_PPI_connectivity_reg_centered_allConds')
    mat_date = '2020-03-09' #Date that matrix regressors were generated
    subs=pd.read_csv(home + '/LS_scripts/LS_sublist.txt', sep=' ', header=None)[0].tolist()


# ## Functions

# ### Binarize convolved data

# In[5]:


#Function to binarize convolved data (converts data above 0 to 1, 0 or below to 0)
def binarize(data_series):
    binarized = []
    for x in range(0,len(data_series)):
        if data_series[x] > 0:
            binarized.append(1)
        elif data_series[x] == 0:
            binarized.append(0)
        elif data_series[x] < 0:
            binarized.append(-1)
        else:
            print("Error")
    return pd.Series(binarized)
            


# ### Parse two-run subs and one-run subs

# In[6]:


one_run = []
two_runs = []
no_runs = []
def parse_runs(sub):
    if user == 'laptop':
        roimean_dir = home
    elif user == 'mrrc':
        roimean_dir = '/mnt/abcd/derivatives/{}/LS_ABCD_368_with_GSR_LP1'.format(sub)
    
    roimean_list=glob(roimean_dir + '/R_{}_ses-baselineYear1Arm1_task-nBack_run-*_bold_bis_matrix_1_roimean.txt'.format(sub))
    
    if len(roimean_list) == 2:
        print('**2** runs for {}'.format(sub))
        two_runs.append(sub)
    elif len(roimean_list) == 1:
        print('*1* run for {}'.format(sub))
        one_run.append(sub)
    else:
        print('Neither 1 nor 2 ROImean files found for {}'.format(sub))
        no_runs.append(sub)
        


# ### Perform OLS PPI Regression
# 
# ##### Per  betaCalc_intraSubPPI.m script (converted)

# In[11]:


# pd.DataFrame(place2b_int_array).to_csv('~/Desktop/test_mat.csv')


# In[62]:


def run_ppi_regression(sub, run):
   #np.shape(roidata1) 
   if run == 1:
       roimean = roimean_scan1
       bin_reg_matrix = bin_reg_matrix1
       conv_reg_matrix = conv_reg_matrix_1mat
   elif run == 2:
       roimean = roimean_scan2
       bin_reg_matrix = bin_reg_matrix2
       conv_reg_matrix = conv_reg_matrix_2mat
   #Z score ROI data
   print('Z-scoring ROI data')
   roidata = pd.DataFrame(roimean).drop(["ROI ", "Unnamed: 369"], axis=1).to_numpy()
   roidata = pd.DataFrame(roimean).drop(["ROI "], axis=1).to_numpy()
   roidata_zscore = zscore(roidata)

   #roidata2 = zscore(roimean_scan2)

   # Multiple Regression
   ncols = roidata_zscore.shape[1]
   output_list = []
   print('Performing PPI Regression')
   int_roi_array = np.empty([368,368])
   ith_roi_array = np.empty([368,368])
   negface0b_int_array = np.empty([368,368])
   neutface0b_int_array = np.empty([368,368])
   posface0b_int_array = np.empty([368,368])
   negface2b_int_array = np.empty([368,368])
   neutface2b_int_array = np.empty([368,368])
   posface2b_int_array = np.empty([368,368])
   place0b_int_array = np.empty([368,368])
   place2b_int_array = np.empty([368,368])

   for i in range(0, roidata_zscore.shape[1]-1): # i is predictor node
       #Compute interaction terms; create matrix with them (node x regressor)
       #Question: we don't want matrix multiplication here, right? That creates a 1x5 vector. ---> No should be 368x368 matrix 
       #Getting around this by multiplying vectors independently and then re-adding to matrix
       negface0b_int = roidata_zscore[:,i]*bin_reg_matrix[:,0]
       neutface0b_int = roidata_zscore[:,i]*bin_reg_matrix[:,1]
       posface0b_int = roidata_zscore[:,i]*bin_reg_matrix[:,2]
       
       negface2b_int = roidata_zscore[:,i]*bin_reg_matrix[:,4]
       neutface2b_int = roidata_zscore[:,i]*bin_reg_matrix[:,5]
       posface2b_int = roidata_zscore[:,i]*bin_reg_matrix[:,6]
       
       place0b_int = roidata_zscore[:,i]*bin_reg_matrix[:,3]
       place2b_int = roidata_zscore[:,i]*bin_reg_matrix[:,7]
       cue_int = roidata_zscore[:,i]*bin_reg_matrix[:,8]
       
       int_reg_mat = np.array([negface0b_int, neutface0b_int, posface0b_int, place0b_int, 
                               negface2b_int, neutface2b_int, posface2b_int, place2b_int, cue_int]).transpose()   
       #Create ones vector(intercept); add axis, transpose so it is a column
       onesv = np.ones(len(roidata_zscore[:,i]))[np.newaxis].transpose()
       # ith node timeseries data; add new axis and transpose so it is column
       i_roi = roidata_zscore[:,i][np.newaxis].transpose()
       # Create "y" matrix with all predictors in --> 
       y_mat_nan = np.concatenate((onesv, i_roi, conv_reg_matrix, int_reg_mat), axis=1)
       #CONVERT NaNs and Infs to 0/very large or very small number
       y_mat = np.nan_to_num(y_mat_nan)
       #pd.DataFrame(y_mat).to_csv(out_dir + '/example{}_defaultNP_ymatrix_scan{}.csv'.format(i,run))
       #y = np.array([np.ones(len(roidata_zscore[:,i])), roidata_zscore[:,i], reg_int.tolist()[0]]).transpose()
       for j in range(0, roidata_zscore.shape[1]-1): # j is target/outcome node
           #Create x matrix with jth node timeseries, add axis, transpose from row to column --> 
           x_mat_nan = roidata_zscore[:,j][np.newaxis].transpose()
           #CONVERT NaNs and Infs to 0/very large or very small number
           x_mat = np.nan_to_num(x_mat_nan)
           #REGRESSION: y = b0 + b1x --> returns x (betas), residuals, rank, and s
           x, residuals, rank, s = np.linalg.lstsq(x_mat, y_mat, rcond=None)
           output = x[0]
           #Extact betas from output
           negface0_beta = output[11]
           neutface0_beta = output[12]
           posface0_beta = output[13]
           place0_beta = output[14]
           negface2_beta = output[15]
           neutface2_beta = output[16]
           posface2_beta = output[17]
           place2_beta = output[18]
           cue_beta = output[19]
           
           #Append output to matrices
           negface0b_int_array[i,j] = negface0_beta
           neutface0b_int_array[i,j] = neutface0_beta
           posface0b_int_array[i,j] = posface0_beta
           place0b_int_array[j,i] = place0_beta
           negface2b_int_array[i,j] = negface2_beta
           neutface2b_int_array[i,j] = neutface2_beta
           posface2b_int_array[i,j] = posface2_beta
           place2b_int_array[j,i] = place2_beta

   #Save out data in CSV
   print('Saving data to CSV!')
   pd.DataFrame(negface0b_int_array).to_csv(out_dir +'/{}_PPI_NegativeFace_0back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(neutface0b_int_array).to_csv(out_dir +'/{}_PPI_NeutralFace_0back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(posface0b_int_array).to_csv(out_dir +'/{}_PPI_PositiveFace_0back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(place0b_int_array).to_csv(out_dir +'/{}_PPI_Place_0back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(negface2b_int_array).to_csv(out_dir +'/{}_PPI_NegativeFace_2back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(neutface2b_int_array).to_csv(out_dir +'/{}_PPI_NeutralFace_2back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(posface2b_int_array).to_csv(out_dir +'/{}_PPI_PositiveFace_2back_Run{}_{}.csv'.format(sub, run, today))
   pd.DataFrame(place2b_int_array).to_csv(out_dir +'/{}_PPI_Place_2back_Run{}_{}.csv'.format(sub, run, today))


# ## Determine whether subject has one or two runs

# In[63]:


for sub in subs:
    parse_runs(sub)
    
# one_run_df = pd.DataFrame(one_run)
# one_run_df.to_csv(home + '/ABCD_Stress_Nback_Subs_OneTaskRun_{}subs.csv'.format(len(one_run)))
# two_run_df = pd.DataFrame(two_runs)
# two_run_df.to_csv(home + '/ABCD_Stress_Nback_Subs_TwoTaskRuns_{}subs.csv'.format(len(two_runs)))
# no_runs_df = pd.DataFrame(no_runs)
# no_runs_df.to_csv(home + '/ABCD_Stress_Nback_Subs_NoTaskRuns_{}subs.csv'.format(len(no_runs)))


# ## Run PPI for 2-run subject list

# In[ ]:


for sub in two_runs:
    try:
        #Glob data with ROI info
        if user == 'laptop':
            roimean_dir = home
        elif user == 'mrrc':
            roimean_dir = '/mnt/abcd/derivatives/{}/LS_ABCD_368_with_GSR_LP1'.format(sub)
        roimean_list=glob(roimean_dir + '/R_{}_ses-baselineYear1Arm1_task-nBack_run-*_bold_bis_matrix_1_roimean.txt'.format(sub))

        #Read in ROI data
        scan1 = roimean_list[0]
        scan2 = roimean_list[1]
        roimean_scan1 = pd.read_csv(scan1, header=0, sep = '\t').replace(0, nan)
        roimean_scan2 = pd.read_csv(scan2, header=0, sep = '\t').replace(0, nan)

        #Read in regressor matrices for SCAN 1 sub-NDARINV1VMB1ZYW_Regressor_Matrix_Block_Scan2_2020-03-03.csv
        reg_matrix_1 = pd.read_csv(reg_dir + '/{}_Regressor_Matrix_Block_Scan1_{}.csv'.format(sub, mat_date), header=0)
        
        #Create overall face regressor column by summing three face conditions
        #reg_matrix_1['face_0back_reg'] = reg_matrix_1['0_back_negface_scan1_reg'] + reg_matrix_1['0_back_neutface_scan1_reg'] + reg_matrix_1['0_back_posface_scan1_reg']  
        #reg_matrix_1['face_2back_reg'] = reg_matrix_1['2_back_negface_scan1_reg'] + reg_matrix_1['2_back_neutface_scan1_reg'] + reg_matrix_1['2_back_posface_scan1_reg']  
        #Select only columns of interest
        #reg_matrix_1 = reg_matrix_1[['face_0back_reg', '0_back_place_scan1_reg', 'face_2back_reg', '2_back_place_scan1_reg', 'cue_scan1_reg']]

        #Read in regressor matrices for SCAN 2
        reg_matrix_2 = pd.read_csv(reg_dir + '/{}_Regressor_Matrix_Block_Scan2_{}.csv'.format(sub, mat_date), header=0) 
        #Create overall face regressor column by summing three face conditions
#         reg_matrix_2['face_0back_reg'] = reg_matrix_2['0_back_negface_scan2_reg'] + reg_matrix_2['0_back_neutface_scan2_reg'] + reg_matrix_2['0_back_posface_scan2_reg']  
#         reg_matrix_2['face_2back_reg'] = reg_matrix_2['2_back_negface_scan2_reg'] + reg_matrix_2['2_back_neutface_scan2_reg'] + reg_matrix_2['2_back_posface_scan2_reg']  
#         #Select only columns of interest
#         reg_matrix_2 =reg_matrix_2[['face_0back_reg', '0_back_place_scan2_reg', 'face_2back_reg', '2_back_place_scan2_reg', 'cue_scan2_reg']]

        #Generate HRF response function from SPM
        hrf = spm_hrf(.8)

        #Create new empty data frames for convolved data
        conv_reg_matrix_1 = pd.DataFrame()
        conv_reg_matrix_2 = pd.DataFrame()

        #Do things (convolve each condition column) Scan 1
        conv_reg_matrix_1['negface_0back_reg_conv'] = convolve(hrf, reg_matrix_1['0_back_negface_scan1_reg'], mode='full')
        conv_reg_matrix_1['neutface_0back_reg_conv'] = convolve(hrf, reg_matrix_1['0_back_neutface_scan1_reg'], mode='full')
        conv_reg_matrix_1['posface_0back_reg_conv'] = convolve(hrf, reg_matrix_1['0_back_posface_scan1_reg'], mode='full')
        conv_reg_matrix_1['negface_2back_reg_conv'] = convolve(hrf, reg_matrix_1['2_back_negface_scan1_reg'], mode='full')
        conv_reg_matrix_1['neutface_2back_reg_conv'] = convolve(hrf, reg_matrix_1['2_back_neutface_scan1_reg'], mode='full')
        conv_reg_matrix_1['posface_2back_reg_conv'] = convolve(hrf, reg_matrix_1['2_back_posface_scan1_reg'], mode='full')
        conv_reg_matrix_1['place_0back_reg_conv'] = convolve(hrf, reg_matrix_1['0_back_place_scan1_reg'], mode='full')
        conv_reg_matrix_1['place_2back_reg_conv'] = convolve(hrf, reg_matrix_1['2_back_place_scan1_reg'], mode='full')
        conv_reg_matrix_1['cue_reg_conv'] = convolve(hrf, reg_matrix_1['cue_scan1_reg'], mode='full')

        #Do things (convolve) Scan 2
        conv_reg_matrix_2['negface_0back_reg_conv'] = convolve(hrf, reg_matrix_2['0_back_negface_scan2_reg'], mode='full')
        conv_reg_matrix_2['neutface_0back_reg_conv'] = convolve(hrf, reg_matrix_2['0_back_neutface_scan2_reg'], mode='full')
        conv_reg_matrix_2['posface_0back_reg_conv'] = convolve(hrf, reg_matrix_2['0_back_posface_scan2_reg'], mode='full')
        conv_reg_matrix_2['negface_2back_reg_conv'] = convolve(hrf, reg_matrix_2['2_back_negface_scan2_reg'], mode='full')
        conv_reg_matrix_2['neutface_2back_reg_conv'] = convolve(hrf, reg_matrix_2['2_back_neutface_scan2_reg'], mode='full')
        conv_reg_matrix_2['posface_2back_reg_conv'] = convolve(hrf, reg_matrix_2['2_back_posface_scan2_reg'], mode='full')
        conv_reg_matrix_2['place_0back_reg_conv'] = convolve(hrf, reg_matrix_2['0_back_place_scan2_reg'], mode='full')
        conv_reg_matrix_2['place_2back_reg_conv'] = convolve(hrf, reg_matrix_2['2_back_place_scan2_reg'], mode='full')
        conv_reg_matrix_2['cue_reg_conv'] = convolve(hrf, reg_matrix_2['cue_scan2_reg'], mode='full')

        #Truncate convolved data to length = 362 ***DO YOU CUT OFF THE ENDS --> look how numpy sets up convolve
        conv_reg_matrix_1 = conv_reg_matrix_1.truncate(after=361)
        conv_reg_matrix_2 = conv_reg_matrix_2.truncate(after=361)
        conv_reg_matrix_1.to_csv(out_dir + '/convolved_mat1_full.csv')
        conv_reg_matrix_2.to_csv(out_dir + '/convolved_mat2.csv')

        #Convert convolved regressors to numpy
        conv_reg_matrix_1mat = conv_reg_matrix_1.to_numpy()
        conv_reg_matrix_2mat = conv_reg_matrix_2.to_numpy()

        #Add columns to data frame with binarized data
        bin_reg_matrix_1 = pd.DataFrame()
        bin_reg_matrix_1['negface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['negface_0back_reg_conv'])
        bin_reg_matrix_1['neutface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['neutface_0back_reg_conv'])
        bin_reg_matrix_1['posface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['posface_0back_reg_conv'])
        bin_reg_matrix_1['negface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['negface_2back_reg_conv'])
        bin_reg_matrix_1['neutface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['neutface_2back_reg_conv'])
        bin_reg_matrix_1['posface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['posface_2back_reg_conv'])
        bin_reg_matrix_1['place_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['place_0back_reg_conv'])
        bin_reg_matrix_1['place_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['place_2back_reg_conv'])
        bin_reg_matrix_1['cue_reg_conv_bin'] = binarize(conv_reg_matrix_1['cue_reg_conv'])

        #Add columns to data frame with binarized data
        bin_reg_matrix_2 = pd.DataFrame()
        bin_reg_matrix_2['negface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_2['negface_0back_reg_conv'])
        bin_reg_matrix_2['neutface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_2['neutface_0back_reg_conv'])
        bin_reg_matrix_2['posface_0back_reg_conv_bin'] = binarize(conv_reg_matrix_2['posface_0back_reg_conv'])
        bin_reg_matrix_2['negface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_2['negface_2back_reg_conv'])
        bin_reg_matrix_2['neutface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_2['neutface_2back_reg_conv'])
        bin_reg_matrix_2['posface_2back_reg_conv_bin'] = binarize(conv_reg_matrix_2['posface_2back_reg_conv'])
        bin_reg_matrix_2['place_0back_reg_conv_bin'] = binarize(conv_reg_matrix_2['place_0back_reg_conv'])
        bin_reg_matrix_2['place_2back_reg_conv_bin'] = binarize(conv_reg_matrix_2['place_2back_reg_conv'])
        bin_reg_matrix_2['cue_reg_conv_bin'] = binarize(conv_reg_matrix_2['cue_reg_conv'])

        #Convert dataframes to matrices
        bin_reg_matrix1= bin_reg_matrix_1.iloc[0:362].to_numpy()
        bin_reg_matrix2= bin_reg_matrix_2.iloc[0:362].to_numpy()

        # RUN FUNCTION TO PERFORM PPI
        run_ppi_regression(sub, 1)
        run_ppi_regression(sub, 2)
    except Exception as inst:
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        # but may be overridden in exception subclasses


# In[65]:


for sub in one_run:
    try:
        print('Working on {}'.format(sub))
        #Glob data with ROI info
        if user == 'laptop':
            roimean_dir = home
        elif user == 'mrrc':
            roimean_dir = '/mnt/abcd/derivatives/{}/LS_ABCD_368_with_GSR_LP1'.format(sub)
        roimean_list=glob(roimean_dir + '/R_{}_ses-baselineYear1Arm1_task-nBack_run-*_bold_bis_matrix_1_roimean.txt'.format(sub))
        #Read in ROI data
        scan1 = roimean_list[0]
        roimean_scan1 = pd.read_csv(scan1, header=0, sep = '\t').replace(0, nan)

        #Read in regressor matrices for SCAN 1
        reg_matrix_1 = pd.read_csv(reg_dir + '/{}_Regressor_Matrix_Block_Scan1_{}.csv'.format(sub, mat_date), header=0)
        #Create overall face regressors 
        reg_matrix_1['face_0back_reg'] = reg_matrix_1['0_back_negface_scan1_reg'] + reg_matrix_1['0_back_neutface_scan1_reg'] + reg_matrix_1['0_back_posface_scan1_reg']  
        reg_matrix_1['face_2back_reg'] = reg_matrix_1['2_back_negface_scan1_reg'] + reg_matrix_1['2_back_neutface_scan1_reg'] + reg_matrix_1['2_back_posface_scan1_reg']  
        #Select only columns of interest
        reg_matrix_1 =reg_matrix_1[['face_0back_reg', '0_back_place_scan1_reg', 'face_2back_reg', '2_back_place_scan1_reg', 'cue_scan1_reg']]

        #Generate HRF response function from SPM
        hrf = spm_hrf(.8)

        #Create new data frames for convolved data
        conv_reg_matrix_1 = pd.DataFrame()

        #Do things (convolve) Scan 1
        conv_reg_matrix_1['face_0back_reg_conv'] = convolve(hrf, reg_matrix_1['face_0back_reg'], mode='full')
        conv_reg_matrix_1['place_0back_reg_conv'] = convolve(hrf, reg_matrix_1['0_back_place_scan1_reg'], mode='full')
        conv_reg_matrix_1['face_2back_reg_conv'] = convolve(hrf, reg_matrix_1['face_2back_reg'], mode='full')
        conv_reg_matrix_1['place_2back_reg_conv'] = convolve(hrf, reg_matrix_1['2_back_place_scan1_reg'], mode='full')
        conv_reg_matrix_1['cue_reg_conv'] = convolve(hrf, reg_matrix_1['cue_scan1_reg'], mode='full')

        #Truncate convolved data to length = 362 ***DO YOU CUT OFF THE ENDS --> look how numpy sets up convolve
        conv_reg_matrix_1 = conv_reg_matrix_1.truncate(after=361)

        #Convert convolved regressors to numpy
        conv_reg_matrix_1mat = conv_reg_matrix_1.to_numpy()

        #Add columns to data frame with binarized data
        bin_reg_matrix_1 = pd.DataFrame()
        bin_reg_matrix_1['face_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['face_0back_reg_conv'])
        bin_reg_matrix_1['place_0back_reg_conv_bin'] = binarize(conv_reg_matrix_1['place_0back_reg_conv'])
        bin_reg_matrix_1['face_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['face_2back_reg_conv'])
        bin_reg_matrix_1['place_2back_reg_conv_bin'] = binarize(conv_reg_matrix_1['place_2back_reg_conv'])
        bin_reg_matrix_1['cue_reg_conv_bin'] = binarize(conv_reg_matrix_1['cue_reg_conv'])

        #Convert dataframes to matrices
        bin_reg_matrix1= bin_reg_matrix_1.iloc[0:362].to_numpy()

        # RUN FUNCTION TO PERFORM PPI
        run_ppi_regression(sub, 1)
        print('      ')
        print('All done!')
        print('      ')
    except Exception as inst:
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        # but may be overridden in exception subclasses


# In[ ]:




