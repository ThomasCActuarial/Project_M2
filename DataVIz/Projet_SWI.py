# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""
import pandas as pd
import sklearn as sk

#%%
df1 = pd.read_csv('data\MENS_SIM2_2010-2019.csv',delimiter=';' )
df2 = pd.read_csv('data\MENS_SIM2_2000-2009.csv',delimiter=';')
df3 = pd.read_csv('data\MENS_SIM2_1990-1999.csv',delimiter=';')
df4 = pd.read_csv('data\MENS_SIM2_1980-1989.csv',delimiter=';')
df5 = pd.read_csv('data\MENS_SIM2_latest-2020-2024.csv',delimiter=';' )
data=pd.concat([df1,df2,df3,df4,df5])
del df1, df2, df3, df4 ,df5
#%%



