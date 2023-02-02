#!/usr/bin/env python
# coding: utf-8

# In[3]:


from glob import glob
import pandas as pd
from datetime import date
import numpy as np
from shutil import copyfile
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)
today = str(date.today())


# In[6]:


user = 'mrrc'

home = ''
bothruns_usable = pd.read_csv(home + '/../subjectlists/run1and2_usable_5.4.20.txt',
                              header=None, sep=' ').rename(columns={0: "subjectkey"})
out = home + '/../LS_PPI_Averaged_reg_centered_allConds'
motion = pd.read_csv(home + '/../All_MotionStats_Scan1Scan2_nBack_2020-03-09.csv',
                     header=0).drop("Unnamed: 0", axis=1).rename(columns={"0": "subjectkey"})
motion['mean_mot'] = motion[['rel_fd_scan1', 'rel_fd_scan2']].mean(axis=1)
subids = bothruns_usable['subjectkey']


# ### Get Average of Matrices for IDs with Two Runs

# In[3]:


conditions = ["NeutralFace_0back", "NeutralFace_2back", "NegativeFace_0back", "NegativeFace_2back",
              "PositiveFace_0back", "PositiveFace_2back", "Place_0back", "Place_2back"]

# Read in Data
#subids = subids[0:30]


def load_data(cond):
    today = str(date.today())
    data_scan = []
    subs_added = []
    for i in range(0, len(subids)):
        sub = subids.sort_values()[i]
        print("Loading {} for {}, {} out of {}".format(sub, cond, i, len(subids)))
        subs_added.append(sub)
        matlist = glob(home + '/*{}*{}*.csv'.format(sub, cond))
        # Generate and load data
        try:
            #Read in data
            mat1 = np.genfromtxt(matlist[0], delimiter=",", skip_header=1)[
                :, 1:369]
            mat2 = np.genfromtxt(matlist[1], delimiter=",", skip_header=1)[
                :, 1:369]
            # Average run 1 and run2 together
            avg_mat = np.mean(np.array([mat1,  mat2]), axis=0)
            mat_t = avg_mat.transpose()
            # Symmetrize matrices by averaging with transpose
            final_mat = np.mean(np.array([avg_mat,  mat_t]), axis=0)
            pd.DataFrame(final_mat).to_csv(
                out + '/{}_{}_FinalAveraged_{}.csv'.format(sub, cond, today))

            f_mat = final_mat.flatten(order='C')
            data_scan.append(f_mat)

        except Exception as inst:
            print("ERROR for {} on condition {}".format(sub, cond))
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)

    return subs_added, data_scan


# In[4]:


print("Working on 0-back")
neg_f0_subs, neg_scan_1f0 = load_data("NegativeFace_0back")
neut_f0_subs, neut_scan_1f0 = load_data("NeutralFace_0back")
pos_f0_subs, pos_scan_1f0 = load_data("PositiveFace_0back")
place0_subs, place0_scan = load_data("Place_0back")

print("Working on 2-back")
neg_f2_subs, neg_scan_1f2 = load_data("NegativeFace_2back")
neut_f2_subs, neut_scan_1f2 = load_data("NeutralFace_2back")
pos_f2_subs, pos_scan_1f2 = load_data("PositiveFace_2back")
place2_subs, place2_scan = load_data("Place_2back")


# In[5]:


# Clean up dataframes
subs_added = neg_f0_subs
negfacef0_df = pd.concat(
    [pd.Series(neg_f0_subs, name='subjectkey'), pd.DataFrame(neg_scan_1f0)], axis=1)
neutfacef0_df = pd.concat(
    [pd.Series(neut_f0_subs, name='subjectkey'), pd.DataFrame(neut_scan_1f0)], axis=1)
posfacef0_df = pd.concat(
    [pd.Series(pos_f0_subs, name='subjectkey'), pd.DataFrame(pos_scan_1f0)], axis=1)
place0_df = pd.concat(
    [pd.Series(place0_subs, name='subjectkey'), pd.DataFrame(place0_scan)], axis=1)

negfacef2_df = pd.concat(
    [pd.Series(neg_f2_subs, name='subjectkey'), pd.DataFrame(neg_scan_1f2)], axis=1)
neutfacef2_df = pd.concat(
    [pd.Series(neut_f2_subs, name='subjectkey'), pd.DataFrame(neut_scan_1f2)], axis=1)
posfacef2_df = pd.concat(
    [pd.Series(pos_f2_subs, name='subjectkey'), pd.DataFrame(pos_scan_1f2)], axis=1)
place2_df = pd.concat(
    [pd.Series(place2_subs, name='subjectkey'), pd.DataFrame(place2_scan)], axis=1)


# In[47]:


def regress_motion(scan):
    print(scan.shape)
    # Merge data and run regression
    subs_df = scan[["subjectkey"]]
    motion_data = pd.merge(subs_df, motion, how="inner", on="subjectkey")
    print(motion_data.shape)

    cov1 = motion_data['mean_mot'].to_numpy().reshape(1, -1)

    # Run regressions
    scan_corr = np.ones((len(scan), len(scan.columns)), dtype='object')

    try:
        print("Starting regressions")
        for i in range(0, len(scan.columns) - 1):
            x = i + 1
            print("Regressing column {} out of {}".format(x, len(scan1.columns)))
            # Scan 1
            col1o = scan.iloc[:, x].to_numpy().reshape(1, -1)
            col1 = add_constant(col1o, prepend=True)
            model1 = OLS(endog=cov1, exog=col1).fit()
            y_predicted1 = model1.predict()
            resid1 = model1.resid
            scan_corr[:, x] = resid1

    except Exception as inst:
        print("ERROR")
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)

    return pd.DataFrame(scan_corr)


# In[48]:

print("Working on negace0")
neg_f0_reg = regress_motion(negfacef0_df.dropna())
print("Working on neutface0")
neut_f0_reg = regress_motion(neutfacef0_df.dropna())
print("Working on posface0")
pos_f0_reg = regress_motion(posfacef0_df.dropna())
print("Working on place0")
place_0_reg = regress_motion(place0_df.dropna())

print("Working on negace2")
neg_f2_reg = regress_motion(negfacef2_df.dropna())
print("Working on neutface2")
neut_f2_reg = regress_motion(neutfacef2_df.dropna())
print("Working on posface2")
pos_f2_reg = regress_motion(posfacef2_df.dropna())
print("Working on place2")
place_2_reg = regress_motion(place2_df.dropna())


# In[49]:


scaler = StandardScaler()
print("Scaling 0-back data")
neg_f0_reg_scale = scaler.fit_transform(
    neg_f0_reg.iloc[:, 1:len(neg_f0_reg.columns)])
neut_f0_reg_scale = scaler.fit_transform(
    neut_f0_reg.iloc[:, 1:len(neut_f0_reg.columns)])
pos_f0_reg_scale = scaler.fit_transform(
    pos_f0_reg.iloc[:, 1:len(pos_f0_reg.columns)])
place_0_reg_scale = scaler.fit_transform(
    place_f0_reg.iloc[:, 1:len(place_0_reg.columns)])
print("Scaling 2-back data")
neg_f2_reg_scale = scaler.fit_transform(
    neg_f2_reg.iloc[:, 1:len(neg_f2_reg.columns)])
neut_f2_reg_scale = scaler.fit_transform(
    neut_f2_reg.iloc[:, 1:len(neut_f2_reg.columns)])
pos_f2_reg_scale = scaler.fit_transform(
    pos_f2_reg.iloc[:, 1:len(pos_f2_reg.columns)])
place_2_reg_scale = scaler.fit_transform(
    place_f2_reg.iloc[:, 1:len(place_2_reg.columns)])


# In[55]:


today = str(date.today())


def reshape_data(df, subs, cond):
    out_dir = out + '/../LS_PPI_connectivity_reg_centered_allConds_MotionRegressed'
    for i in range(0, len(df)):
        row = df[i, 0:135425]
        sub = pd.Series(subs_added)[i]
        print("Working on {}".format(sub))
        mat = row.reshape(368, 368, order='C')
        mat_t = mat.transpose()
        finalmat = np.mean(np.array([mat,  mat_t]), axis=0)
        pd.DataFrame(finalmat).to_csv(
            out_dir + '/{}_{}_FInalAveraged_MotionCorrected_{}.csv'.format(sub, cond, today))


# In[56]:


conditions = ["NeutralFace_0back", "NeutralFace_2back", "NegativeFace_0back", "NegativeFace_2back",
              "PositiveFace_0back", "PositiveFace_2back", "Place_0back", "Place_2back"]

print("Reshaping negface 0back")
reshape_data(neg_f0_reg_scale, subs_added, "NegativeFace_0back")
print("Reshaping neutface 0back")
reshape_data(neut_f0_reg_scale, subs_added, "NeutralFace_0back")
print("Reshaping posface 0back")
reshape_data(pos_f0_reg_scale, subs_added, "PositiveFace_0back")
print("Reshaping place 0back")
reshape_data(place_0_reg_scale, subs_added, "Place_0back")

print("Reshaping negface 2back")
reshape_data(neg_f2_reg_scale, subs_added, "NegativeFace_2back")
print("Reshaping neutface 2back")
reshape_data(neut_f2_reg_scale, subs_added, "NeutralFace_2back")
print("Reshaping posface 0back")
reshape_data(pos_f2_reg_scale, subs_added, "PositiveFace_2back")
print("Reshaping place 0back")
reshape_data(place_2_reg_scale, subs_added, "Place_2back")


# In[ ]:


# conditions = ["PPI_face2", "PPI_face0", "PPI_place2", "PPI_place0"]
# subs = []
# for x in range(0, len(subids)): ##len(bothruns_usable)
#     sub = subids.sort_values()[x]
#     subs.append(sub)
#     print('Working on {}, sub {} of {}'.format(sub, x, len(subids)))
#     mot1 = motion[motion.subjectkey == sub]['rel_fd_scan1']
#     mot2 = motion[motion.subjectkey == sub]['rel_fd_scan2']
#     for y in range(0, len(conditions)):
#         cond = conditions[y]
#         print(cond)
#         matlist = glob(home + '/' + cond + '/*{}*.csv'.format(sub))
#         try:
#             mat1 = np.genfromtxt(matlist[0], delimiter=",", skip_header=1)[:,1:369]
#             mat2 = np.genfromtxt(matlist[1], delimiter=",", skip_header=1)[:,1:369]

#             mot_mat1 = regress_motion(mat1, mot1)
#             mot_mat2 = regress_motion(mat2, mot2)

#             mean_mat = np.mean( np.array([ mot_mat1, mot_mat2 ]), axis=0 )

#             #Average with transpose for final data
#             mean_mat_t = mean_mat.transpose()
#             finmat = np.mean( np.array([ mean_mat, mean_mat_t ]), axis=0 )

#             pd.DataFrame(meanmat).to_csv(out + '/{}/{}_Final_Matrix_Averaged_MotionRegressed_{}.csv'.format(cond, sub, today),
#                        index=False)
#         except Exception as inst:
#             print("ERROR for {} on condition {}".format(sub, cond))
#             print(type(inst))    # the exception instance
#             print(inst.args)     # arguments stored in .args
#             print(inst)          # __str__ allows args to be printed directly,
#             # but may be overridden in exception subclasses


# In[ ]:
