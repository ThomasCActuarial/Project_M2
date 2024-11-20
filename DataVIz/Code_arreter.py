# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:51:23 2024

@author: yacko
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:51:16 2024

@author: yacko
"""
import time 
import overpy
import csv
import requests

import pandas as pd


import folium


Commerce=[]

#%%
# chemin du csv
chemin_fichier_csv = "000838(1).csv"


# permet une meilleur présition que les nom donner par le fichier de base a cause de la difference entre espace et tiret 

def Cinsee(code_insee):
    code_insee=str(code_insee)
    if len(code_insee)<5:
        code_insee = "0"+code_insee
    
    time.sleep(0.03) #NE PAS TOUCHER A CAUSE DES LIMITES DE L'API 
    #SINON ban connexion => EViter les gars
    url = f'https://geo.api.gouv.fr/communes?code={code_insee}&fields=code,nom,population,surface,zone,centre,contour'
    
    reponse = requests.get(url)
    if reponse.status_code == 200:
        s = reponse.json()
        ret =[ [s[0]['surface'], s[0]['population'],s[0]['centre']['coordinates'],s[0]['zone']]]
        
    return( ret)


print()

# %%

CAT= pd.read_csv("data/CATNAT.csv")

# commune_a_rechercher = "Saint-Cyr"
# resultats = trouver_commerces_dans_commune(commune_a_rechercher)
# %%
resp=dict()
df_restreint = CAT
for x in  pd.unique( df_restreint["cod_commune"]) :
    resp[x]=Cinsee( x)

# %%

