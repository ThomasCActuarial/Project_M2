
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
# On limite les donnée qu'on utilise
CAT = CAT.drop(columns=['cod_nat_catnat', 'num_risque_jo', 'dat_pub_jo', 'dat_maj'])
Commune = Commune.drop(columns=[ 'id', 'dep', 'reg', 'xcl2154', 'ycl2154',        ])
data = data.drop(columns=["SSWI1_MENS","SSWI6_MENS","SSWI12_MENS"])
dates = pd.date_range(end="2022-12-31", periods=60, freq='M')
CAT['dat_fin'] = pd.to_datetime(CAT['dat_fin'])
CAT['dat_deb'] = pd.to_datetime(CAT['dat_deb'])
CAT['dat_pub_arrete'] = pd.to_datetime(CAT['dat_pub_arrete'])
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')
data['SPEI_1'] = data['PRELIQ_MENS']-data['ETP_MENS']
data["KEY"] = data['LAMBY'].apply(str).str.cat( data['LAMBX'].apply(str), sep=",")
# On garde que les donnée que l'on veux garder
CAT = CAT[CAT['dat_fin']>=min(dates) ]
dates = pd.DataFrame(dates)
# On crée notre base de donnée
Communes = Commune.merge(dates, how='cross')
Communes = Communes.rename(columns={0: "DATE"})
Communes["dry"] = 0

#%% On implimante nos catasphrophe , code assez long ~30mn a tournée
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

#%% Recuper les résultat précedent pour éviter d'avoir a refaire tourner le code
Communes=pd.read_pickle("CommunesCopy")


#%%

data2=data[data.DATE > "2015-12-31"]
geometry = [Point(xy) for xy in zip(data2["LAMBX"], data2["LAMBY"])]

# Étape 2: Créer un GeoDataFrame avec le CRS Lambert 93
gdf = gpd.GeoDataFrame(data2, geometry=geometry, crs="EPSG:2154")  # EPSG:2154 correspond à Lambert 93

# Étape 3: Reprojeter en WGS 84 (EPSG:4326)
gdf_wgs84 = gdf.to_crs(epsg=4326)  # EPSG:4326 correspond à WGS 84

#%%

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

#%%
# le petit code de Christopher  Qu'es ce que ce code vient faire là?  
import pandas as pd

# Création du DataFrame à partir des données
dataChir = {
    "LAMBX": [600, 760, 840, 920, 1000, 1080, 1160, 1240],
    "LAMBY": [24010, 23610, 23930, 24090, 24170, 23530, 23850, 23770]
}
df = pd.DataFrame(data)
print(df)

from pyproj import Transformer

# Initialisation du transformateur Lambert 93 -> WGS84
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

# Conversion des coordonnées
df["Longitude"], df["Latitude"] = transformer.transform(df["LAMBX"], df["LAMBY"])
print(df)


#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the features and target variable
features = ['PRENEI_MENS', 'PRELIQ_MENS', 'PRETOTM_MENS', 'T_MENS', 'EVAP_MENS', 'ETP_MENS', 'SWI_MENS']
X = w[features]  # Features
y = w['dry']      # Target variable (binary: 1 or 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

#%%

def plot_feature_importances(coefficients, feature_names, model_name):
    """
    Plots the feature importances based on the coefficients of the model.
    
    Parameters:
    - coefficients: Array of model coefficients (absolute values for importance).
    - feature_names: List of feature names corresponding to the coefficients.
    - model_name: String name of the model for labeling the plot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Create a DataFrame for coefficients and feature names
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    }).sort_values(by='Importance', ascending=False)
    
    # Plot the importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title(f'Feature Importances - {model_name}')
    plt.xlabel('Coefficient Magnitude (Feature Importance)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# Call the function for your logistic regression model
lr_model = model  # Your logistic regression model
plot_feature_importances(lr_model.coef_[0], features, 'Logistic Regression')