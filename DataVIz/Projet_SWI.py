
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""

#utilsier research pour les hyperparametrage ( grid search /  hyper parametrag )
#analyse de erreur est importante
# Rapport environ 10 pages
# Pas de presentation des algo mais justifier le choix des modèles

import pandas as pd
import sklearn as sk
import numpy as np
import math
import geopandas as gpd
import plotly.express as px
from shapely import wkt
from shapely.geometry import Point
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
Communes = Communes.rename(columns={0: "DATE"})
Communes["dry"] = 0

#%% On implimante nos catasphrophe
i=0;
for code in pd.unique(Communes["codgeo"]):
    i+=1
    # Filter CAT for the current code
    CATNAT = CAT[CAT["cod_commune"] == code]
    if i%500==0:
        print(i)
    if len(CATNAT) > 0:
        sample = Communes[Communes["codgeo"] == code]

        for idx_catnat, row_catnat in CATNAT.iterrows():
            # Check if sample['DATE'] is within the date range [dat_deb, dat_fin]
            condition = sample["DATE"].between(row_catnat["dat_deb"], row_catnat["dat_fin"])

            # Update 'dry' to 1 where condition is True
            Communes.loc[sample[condition].index, "dry"] = 1

#%% Recuper les résultat précedent
Communes=pd.read_pickle("CommunesCopy")


#%%

def create_annual_dry_map(data, geometry_col, date_col, dry_col, title="Dry Zones by Year"):
    """
    Crée une carte interactive annuelle avec des zones `dry=1` en rouge, et un slider pour l'année.

    Args:
    - data (pd.DataFrame or gpd.GeoDataFrame): Tableau contenant les géométries, les dates et `dry`.
    - geometry_col (str): Colonne contenant les géométries (zones).
    - date_col (str): Colonne contenant les dates (format '%Y-%m').
    - dry_col (str): Colonne contenant les valeurs `dry` (0 ou 1).
    - title (str): Titre de la carte.
    """
    # S'assurer que le tableau est un GeoDataFrame
    if not isinstance(data, gpd.GeoDataFrame):
        data = gpd.GeoDataFrame(data, geometry=data[geometry_col])

    # S'assurer que la colonne de dates est au format datetime
    data[date_col] = pd.to_datetime(data[date_col], format="%Y-%m")  # Convertir en datetime
    data["year"] = data[date_col].dt.year  # Extraire l'année

    # Filtrer uniquement les zones où dry=1 pour chaque géométrie et année
    annual_data = data[data[dry_col] == 1].copy()

    # Convertir les géométries en GeoJSON
    geojson = annual_data.geometry.__geo_interface__

    # Ajouter un identifiant unique pour chaque géométrie
    annual_data["id"] = annual_data.index.astype(str)

    # Créer une carte interactive avec Plotly Express
    fig = px.choropleth_mapbox(
        annual_data,
        geojson=geojson,
        locations="id",  # Utiliser l'identifiant unique
        color="year",  # Utiliser l'année comme gradient de couleur
        hover_name="year",
        animation_frame="year",  # Slider sur l'année
        mapbox_style="carto-positron",
        title=title,
        center={"lat": 46.5, "lon": 2.0},  # Centré sur la France (par défaut)
        zoom=5,
    )

    # Ajuster la hauteur
    fig.update_layout(height=700)

    # Afficher la carte
    fig.show()




Communes2 = Communes[Communes["dry"]==1]
create_annual_dry_map(Communes2,"geometry","DATE","dry")


#%%

data2=data[data.DATE > "2015-12-31"]
geometry = [Point(xy) for xy in zip(data2["LAMBX"], data2["LAMBY"])]

# Étape 2: Créer un GeoDataFrame avec le CRS Lambert 93
gdf = gpd.GeoDataFrame(data2, geometry=geometry, crs="EPSG:2154")  # EPSG:2154 correspond à Lambert 93

# Étape 3: Reprojeter en WGS 84 (EPSG:4326)
gdf_wgs84 = gdf.to_crs(epsg=4326)  # EPSG:4326 correspond à WGS 84









