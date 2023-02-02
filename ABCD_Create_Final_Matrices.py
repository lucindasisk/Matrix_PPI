#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import pandas as pd
from datetime import date
import numpy as np
from shutil import copyfile
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)
today = str(date.today())


# In[ ]:


user = 'mrrc'
home = ''
bothruns_usable = pd.read_csv(
    home + '/../subjectlists/run1and2_usable_5.4.20.txt', header=None, sep=' ')
out = home + '/../LS_PPI_Averaged_reg_centered'


# ### Get Average of Matrices for IDs with Two Runs

# In[ ]:


conditions = ["PPI_face2", "PPI_face0", "PPI_place2", "PPI_place0"]
for x in range(0, len(bothruns_usable)):  # len(bothruns_usable)
    sub = bothruns_usable[0].tolist()[x]
    print('Working on {}, sub {} of {}'.format(sub, x, len(bothruns_usable)))
    for y in range(0, len(conditions)):
        cond = conditions[y]
        print(cond)
        matlist = glob(home + '/' + cond + '/*{}*.csv'.format(sub))
        try:
            mat1 = np.genfromtxt(matlist[0], delimiter=",", skip_header=1)[
                :, 1:369]
            mat2 = np.genfromtxt(matlist[1], delimiter=",", skip_header=1)[
                :, 1:369]
            meanmat = np.mean(np.array([mat1, mat2]), axis=0)
            pd.DataFrame(meanmat).to_csv(out + '/{}/{}_Final_Matrix_Averaged_{}.csv'.format(cond, sub, today),
                                         index=False)
        except Exception as inst:
            print("ERROR for {} on condition {}".format(sub, cond))
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            # but may be overridden in exception subclasses


# In[18]:


print('DONE')


# In[ ]:
