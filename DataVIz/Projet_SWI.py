
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""

#utilsier research pour les hyperparametrage ( grid search /  hyper parametrag )
#analyse de erreur est importante
# Rapport environ 10 pages
# Pas de presentation des algo mais justifier le choix des modèles
from shapely.geometry import Point
import pandas as pd
import sklearn as sk
import numpy as np
import math
import geopandas as gpd

#
#https://www.data.gouv.fr/fr/datasets/contours-des-communes-de-france-simplifie-avec-regions-et-departement-doutre-mer-rapproches/
#

#%% Lecture des données
CAT= pd.read_csv("data/CATNAT.csv")
Commune = gpd.read_file("data/commune.json" )    
df4 = pd.read_csv('data\MENS_SIM2_2010-2019.csv',delimiter=';' )
df5 = pd.read_csv('data\MENS_SIM2_latest-2020-2024.csv',delimiter=';' )

df3 = pd.read_csv('data\MENS_SIM2_2000-2009.csv',delimiter=';')
df2 = pd.read_csv('data\MENS_SIM2_1990-1999.csv',delimiter=';')
df1 = pd.read_csv('data\MENS_SIM2_1980-1989.csv',delimiter=';')
data=pd.concat([df1,df2,df3,df4,df5] ,ignore_index=True)
del df1, df2, df3, df4 ,df5
#%% On limite les donnée qu'on utilise
CAT = CAT.drop(columns=['cod_nat_catnat', 'num_risque_jo', 'dat_pub_jo', 'dat_maj'])
Commune = Commune.drop(columns=[ 'id', 'dep', 'reg', 'xcl2154', 'ycl2154',        ])
data = data.drop(columns=["SSWI1_MENS","SSWI6_MENS","SSWI12_MENS"])
dates = pd.date_range(end="2022-12-31", periods=60, freq='M')

 #%%
CAT['dat_fin'] = pd.to_datetime(CAT['dat_fin'])
CAT['dat_deb'] = pd.to_datetime(CAT['dat_deb'])
CAT['dat_pub_arrete'] = pd.to_datetime(CAT['dat_pub_arrete'])
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')
data['SPEI_1'] = data['PRELIQ_MENS']-data['ETP_MENS']
data["KEY"] = data['LAMBY'].apply(str).str.cat( data['LAMBX'].apply(str), sep=",")

#%% On garde que les donnée que l'on veux garder

CAT = CAT[CAT['dat_fin']>=min(dates) ]
dates = pd.DataFrame(dates)
#%% On crée notre base de donnée

Communes = Commune.merge(dates, how='cross')

Communes["dry"] = 0

#%% On implimante nos catasphrophe
      
for code in pd.unique(Communes["codgeo"]):
    # Filter CAT for the current code
    CATNAT = CAT[CAT["cod_commune"] == code]
    
    if len(CATNAT) > 0:
        # Filter Communes for the current code
        sample = Communes[Communes["codgeo"] == code]
        
        for idx_sample, row_sample in sample.iterrows():
            # Check each row in CATNAT for date range
            for idx_catnat, row_catnat in CATNAT.iterrows():
                if row_catnat["dat_deb"] <= row_sample["DATE"] <= row_catnat["dat_fin"]:
                    Communes.loc[idx_sample, "dry"] = 1  # Update 'dry' to 1
       
        






