# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""
import pandas as pd
import sklearn as sk
import numpy as np
from datetime import datetime 

#%%
df5 = pd.read_csv('data\MENS_SIM2_latest-2020-2024.csv',delimiter=';' )
df4 = pd.read_csv('data\MENS_SIM2_2010-2019.csv',delimiter=';' )
df3 = pd.read_csv('data\MENS_SIM2_2000-2009.csv',delimiter=';')
df2 = pd.read_csv('data\MENS_SIM2_1990-1999.csv',delimiter=';')
df1 = pd.read_csv('data\MENS_SIM2_1980-1989.csv',delimiter=';')

#del df1, df2, df3, df4 ,df5
#%%

data=pd.concat([df1,df2,df3,df4,df5] ,ignore_index=True)

data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')
data=data.sort_values(by=['KEY', 'DATE']).reset_index(drop=True)


data['SPEI_1'] = data['PRELIQ_MENS']-data['ETP_MENS']

data["KEY"] = data['LAMBY'].apply(str).str.cat( data['LAMBX'].apply(str))

#%%

grouped_data = data.groupby(['KEY'])

        
        
        
        
        
        
#%%

dataTest = data[ data["DATE"] !=datetime(2024,10,1,00,00,00)]


def two_largest_last_300(group):
    # Vérifier qu'il y a au moins 300 observations
    if len(group) >= 300:
        # Extraire les 300 dernières valeurs
        last_300_values = group['SSWI12_MENS'].iloc[-300:]
        # Trouver les 2 plus grandes valeurs
        top_2 = last_300_values.nlargest(2)
        return top_2
    else:
        return pd.Series([np.nan, np.nan], index=['top_1', 'top_2'])

# Appliquer la fonction à chaque groupe
result = dataTest.groupby('KEY').apply(two_largest_last_300)

        
        
        
        
        
        
        
        
        
        
        
        
        