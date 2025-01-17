
"""
Created on Mon Nov 11 11:56:50 2024

@author: yacko
"""

#utilsier research pour les hyperparametrage ( grid search /  hyper parametrag )
#analyse de erreur est importante
# Rapport environ 10 pages
# Pas de presentation des algo mais justifier le choix des modèles
import pandas as pd
import numpy as np
import sklearn as sk
import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point
from scipy.spatial import cKDTree
import pickle
#
#https://www.data.gouv.fr/fr/datasets/contours-des-communes-de-france-simplifie-avec-regions-et-departement-doutre-mer-rapproches/
#

# Lecture des données
CAT= pd.read_csv("data/CATNAT.csv")
Commune = gpd.read_file("data/commune.json" )    
df4 = pd.read_csv('data\MENS_SIM2_2010-2019.csv',delimiter=';' )
df5 = pd.read_csv('data\MENS_SIM2_latest-2020-2024.csv',delimiter=';' )

df3 = pd.read_csv('data\MENS_SIM2_2000-2009.csv',delimiter=';')
df2 = pd.read_csv('data\MENS_SIM2_1990-1999.csv',delimiter=';')
df1 = pd.read_csv('data\MENS_SIM2_1980-1989.csv',delimiter=';')
data=pd.concat([df1,df2,df3,df4,df5] ,ignore_index=True)
del df1, df2, df3, df4 ,df5
# On limite les donnée qu'on utilise
CAT = CAT.drop(columns=['cod_nat_catnat', 'num_risque_jo', 'dat_pub_jo', 'dat_maj'])
Commune = Commune.drop(columns=[ 'id', 'dep', 'xcl2154', 'ycl2154'])
data = data.drop(columns=["SSWI1_MENS","SSWI6_MENS","SSWI12_MENS"])
dates = pd.date_range(end="2022-12-31", periods=60, freq='M')
CAT['dat_fin'] = pd.to_datetime(CAT['dat_fin'])
CAT['dat_deb'] = pd.to_datetime(CAT['dat_deb'])
CAT['dat_pub_arrete'] = pd.to_datetime(CAT['dat_pub_arrete'])
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')

data["KEY"] = data['LAMBY'].apply(str).str.cat( data['LAMBX'].apply(str), sep=",")
# On garde que les donnée que l'on veux garder
CAT = CAT[CAT['dat_fin']>=min(dates) ]
dates = pd.DataFrame(dates)
# On crée notre base de donnée
Communes = Commune.merge(dates, how='cross')
Communes = Communes.rename(columns={0: "DATE"})
Communes["dry"] = 0

#On implimante nos catasphrophe , code assez long ~30mn a tournée
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

#Recuper les résultat précedent pour éviter d'avoir a refaire tourner le code
#Communes=pd.read_pickle("CommunesCopy")
#%Communes=Communes.rename(columns= {"geometry" : "formeCommune" })



data2=data[data.DATE > "2015-12-31"]
geometry = [Point(xy) for xy in zip(data2["LAMBX"], data2["LAMBY"])]

# Étape 2: Créer un GeoDataFrame avec le CRS Lambert 93
gdf = gpd.GeoDataFrame(data2, geometry=geometry, crs="EPSG:2154")  # EPSG:2154 correspond à Lambert 93

# Étape 3: Reprojeter en WGS 84 (EPSG:4326)
gdf_wgs84 = gdf.to_crs(epsg=4326)  # EPSG:4326 correspond à WGS 84



def nearest_temperature_by_time(ville, temperature, date_col_ville, date_col_temp):
    """
    Associe chaque Polygon de `ville` au point de `temperature` le plus proche,
    en tenant compte de la proximité spatiale et du même mois/année.
    """
    # Extraire année et mois
    ville["year_month"] = pd.to_datetime(ville[date_col_ville]).dt.to_period("M")
    temperature["year_month"] = pd.to_datetime(temperature[date_col_temp]).dt.to_period("M")

    # Préparer un GeoDataFrame de résultats
    results = []

    # Pour chaque groupe de `year_month` dans `ville`
    for period, ville_group in ville.groupby("year_month"):
        # Filtrer temperature pour le même mois et année
        temp_group = temperature[temperature["year_month"] == period]
        
        if temp_group.empty:
            # Si pas de correspondance temporelle, ignorer ce groupe
            continue

        # Filtrer les géométries non valides dans `ville_group` et `temp_group`
        ville_group = ville_group[ville_group.geometry.notnull()]
        temp_group = temp_group[temp_group.geometry.notnull()]

        temp_group = temp_group[temp_group.geometry.geom_type == "Point"]

        # Calculer les centroïdes pour les Polygons dans `ville_group`
        ville_group = ville_group.copy()
        ville_group["centroid"] = ville_group.geometry.centroid

        # Vérifier si les groupes sont toujours non vides après le filtrage
        if ville_group.empty or temp_group.empty:
            continue

        # Construire un arbre spatial pour les points de `temp_group`
        temp_coords = list(zip(temp_group.geometry.x, temp_group.geometry.y))
        temp_tree = cKDTree(temp_coords)

        # Obtenir les coordonnées des centroïdes de `ville_group`
        ville_coords = list(zip(ville_group["centroid"].x, ville_group["centroid"].y))

        # Trouver les plus proches voisins
        distances, indices = temp_tree.query(ville_coords, k=1)

        # Enregistrer les résultats pour ce groupe
        ville_group["nearest_temp_index"] = temp_group.index.values[indices]
        ville_group["distance_to_temp"] = distances

        results.append(ville_group)

    # Combiner tous les résultats
    enriched_ville = pd.concat(results, ignore_index=True)

    # Fusionner toutes les variables de température dans le GeoDataFrame enrichi
    enriched_ville = enriched_ville.merge(
        temperature, 
        left_on="nearest_temp_index", 
        right_index=True, 
        suffixes=("_ville", "_temp")
    )

    return enriched_ville


w = nearest_temperature_by_time(Communes,gdf_wgs84,"DATE","DATE")
w = w.drop(columns=['SPI1_MENS','SPI3_MENS', 'SPI6_MENS', 'SPI12_MENS', 'SSWI3_MENS', 'ECOULEMENT_MENS','KEY','year_month_ville', 'centroid', 'nearest_temp_index','distance_to_temp', 'LAMBX', 'LAMBY'])
data = w
data=data.drop(columns=['geometry_temp'])

#%%

for var in list(globals().keys()):  # Use list() to avoid runtime modification errors
    if var != "data" and not var.startswith("__"):  # Keep `w` and special variables like __name__
        del globals()[var]


