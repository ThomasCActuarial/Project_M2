
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
from datetime import datetime 

import math
import geopandas as gpd
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
 
    

 
from lambert import Lambert93, convertToWGS84Deg   
  
#%% 
def coordone2(xy):
    
    cord=xy.split(",")     
    x1= int(cord[0])
    y1= int(cord[1])
    cord = Point(x1,y1)
    return cord
#%%
data["point"] = data["KEY"].apply(coordone2)
        
      
        
#%%        
        


print(str(Lambert93.n()))
pt = convertToWGS84Deg(780886, 6980743, Lambert93)
print("Point latitude:" + str(pt.getY()) + " longitude:" + str(pt.getX()))        