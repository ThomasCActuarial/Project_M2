
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""

#utilsier research pour les hyperparametrage ( grid search /  hyper parametrag )
#analyse de erreur est importante
# Rapport environ 10 pages
# Pas de presentation des algo mais justifier le choix des modèles
import shapefile
import pandas as pd
import sklearn as sk
import numpy as np
from datetime import datetime 
import math
import pyproj as p
#%%
df4 = pd.read_csv('data\MENS_SIM2_2010-2019.csv',delimiter=';' )
df5 = pd.read_csv('data\MENS_SIM2_latest-2020-2024.csv',delimiter=';' )

df3 = pd.read_csv('data\MENS_SIM2_2000-2009.csv',delimiter=';')
df2 = pd.read_csv('data\MENS_SIM2_1990-1999.csv',delimiter=';')
df1 = pd.read_csv('data\MENS_SIM2_1980-1989.csv',delimiter=';')

#del df1, df2, df3, df4 ,df5
    #%%

data=pd.concat([df1,df2,df3,df4,df5] ,ignore_index=True)
data = data.drop(columns=["SSWI1_MENS","SSWI6_MENS","SSWI12_MENS"])


  #%%

data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')


data['SPEI_1'] = data['PRELIQ_MENS']-data['ETP_MENS']

data["KEY"] = data['LAMBY'].apply(str).str.cat( data['LAMBX'].apply(str), sep=",")
data=data.sort_values(by=['KEY', 'DATE']).reset_index(drop=True)
#%%

grouped_data = data.groupby(['KEY'])
        
#%%

dataTest = data[ data["DATE"] !=datetime(2024,10,1,00,00,00)]



shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(shuffled_data) * 0.7)
train_data = shuffled_data[:split_index]
test_data = shuffled_data[split_index:]

#%%

sk.linear_model(    )

#%%





#%%
        




def distance_haversine(coord1, coord2):
    R = 6371.0  
    
    # Convertir les degrés en radians
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    
    # Différences des coordonnées
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Formule haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def coordonnee_plus_proche(liste_coordonnees, coordonnee_cible):
    coordonnee_proche = min(liste_coordonnees, key=lambda c: distance_haversine(c, coordonnee_cible))
    return coordonnee_proche
 
    
inProj = p.CRS('epsg:2154')
outProj =p.CRS('epsg:4326')
transformer = p.Transformer.from_crs(inProj, outProj)

transformer.transform(52.067567, 5.068913)    
    
 
    
 
    
 
def coordone2(xy,transformer):
    
    cord=xy.split(",")     
    x1= int(cord[0])
    y1= int(cord[1])
    x2,y2 = transformer.transform(x1,y1)
    cord = str(x2) + "," +str(y2)
    return cord

#w = data["KEY"].apply(coordone2)
        
        
        
        
        
        