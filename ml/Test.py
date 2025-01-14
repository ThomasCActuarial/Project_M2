# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import joblib
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = gpd.read_file("output_0.gpkg")

# Importer les bibliothèques nécessaires


# Supposons que tu aies un DataFrame 'df' avec tes données
# data features = ['PRENEI_MENS', 'PRELIQ_MENS', 'PRETOTM_MENS', 'T_MENS', 'EVAP_MENS', 'ETP_MENS', 'SWI_MENS']
features = [ 'T_MENS', 'SWI_MENS']
target = 'dry'

# Séparer les features et la cible
X = data[features]
y = data[target]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle de forêt aléatoire
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
filename = 'random_forest_model.pkl'
joblib.dump(model, filename)
print(f"Le modèle a été sauvegardé sous le nom {filename}")

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
#%%
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# Charger les données GeoPackage
file_path = 'output_0.gpkg'


# Vérifier et définir un CRS si nécessaire
if data.crs is None:
    data = data.set_crs("EPSG:4326")  # Définir un CRS par défaut
else:
    data = data.to_crs("EPSG:4326")  # Transformer en EPSG:4326 si nécessaire

# Simplifier les géométries pour améliorer les performances
data['geometry'] = data['geometry'].simplify(tolerance=0.01, preserve_topology=True)

# Configuration de l'application Streamlit
st.title("Visualisation des données sur une carte de la France")
st.markdown("Sélectionnez une variable à visualiser sur la carte :")

# Liste des variables disponibles
variables = {
    "Humidité du sol (SWI_MENS)": "SWI_MENS",
    "Température moyenne (T_MENS)": "T_MENS",
    "Précipitations liquides (PRELIQ_MENS)": "PRELIQ_MENS",
}

# Sélectionner une variable
selected_variable = st.selectbox("Choisissez une variable :", options=list(variables.keys()))
variable_column = variables[selected_variable]

# Normaliser la variable sélectionnée pour la visualisation
data['normalized'] = (
    (data[variable_column] - data[variable_column].min()) /
    (data[variable_column].max() - data[variable_column].min())
)

# Créer une carte avec Folium
m = folium.Map(location=[46.603354, 1.888334], zoom_start=5)

# Ajouter les polygones colorés en fonction de la variable sélectionnée
folium.Choropleth(
    geo_data=data,
    data=data,
    columns=['codgeo', 'normalized'],
    key_on='feature.properties.codgeo',
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f"{selected_variable} (normalisé)"
).add_to(m)

# Afficher la carte dans Streamlit
st_folium(m, width=700, height=500)
