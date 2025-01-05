import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = data  # Assignation initiale

# Suppression des colonnes inutiles
#df = df.drop(columns=['geometry_ville', 'year_month_temp', 'DATE_ville', 'DATE_temp', 'codgeo', 'libgeo'])

# Splitting features and target
X = data[['month', 'PRENEI_MENS', 'PRETOTM_MENS', 'T_MENS',
          'EVAP_MENS', 'ETP_MENS', 'PE_MENS', 'SWI_MENS', 'DRAINC_MENS', 'RUNC_MENS']]
y = data['dry']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Testing different class weights
weights = [3,4,5,6]  # Vous pouvez ajouter d'autres poids ici si nécessaire
confusion_matrices = {}

for weight in weights:
    class_weight = {0: 1, 1: weight}
    model = LogisticRegression(random_state=42, class_weight=class_weight, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrices[weight] = confusion_matrix(y_test, y_pred)

# Plotting confusion matrices
fig, axes = plt.subplots(1, len(weights), figsize=(10 * len(weights), 6))

if len(weights) == 1:
    axes = [axes]  # Pour gérer le cas avec une seule matrice de confusion

for i, weight in enumerate(weights):
    cm = confusion_matrices[weight]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f'Weight: {weight}')

plt.tight_layout()
plt.show()




#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
import matplotlib.pyplot as plt

# Exemple de données (remplacez par votre dataset réel)
# Supposons que `data` soit préchargé avec des colonnes : ['codgeo', 'dry', 'DATE_temp', ...]
df = data.drop(columns=['geometry_ville', 'year_month_temp', 'DATE_ville', 'libgeo'])

# Fonction pour créer des séquences
def create_sequences(data, seq_length=10):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length, :].drop(columns=['dry']).values
        label = data.iloc[i+seq_length]['dry']
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Regrouper les données par `codgeo`
grouped_data = data.groupby('codgeo')

sequences = []
labels = []

for name, group in grouped_data:
    group = group.sort_values(by='DATE_temp')  # Trier par date
    seq, lbl = create_sequences(group)
    sequences.append(seq)
    labels.append(lbl)

# Convertir les listes en tableaux NumPy
X = np.vstack(sequences)
y = np.concatenate(labels)

# Diviser les données en ensembles d'entraînement et de test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normaliser les caractéristiques numériques (exclure `codgeo` et `DATE_temp`)
scaler = tf.keras.layers.Normalization(axis=-1)
scaler.adapt(X_train.reshape(-1, X_train.shape[-1])[:, 2:])  # Exclure `codgeo` et `DATE_temp`

X_train_scaled = scaler(X_train.reshape(-1, X_train.shape[-1])[:, 2:]).numpy().reshape(X_train.shape[0], X_train.shape[1], -1)
X_test_scaled = scaler(X_test.reshape(-1, X_test.shape[-1])[:, 2:]).numpy().reshape(X_test.shape[0], X_test.shape[1], -1)

# Calculer les poids de classes pour l'équilibrage
class_weights = {0: (1 / np.sum(y_train == 0)) * len(y_train) / 2.0,
                 1: (1 / np.sum(y_train == 1)) * len(y_train) / 2.0}

# Définir le modèle GRU
def build_gru_model(input_shape, gru_units=32):
    inputs = Input(shape=input_shape)
    gru_output = GRU(gru_units, return_sequences=False)(inputs)
    outputs = Dense(1, activation='sigmoid')(gru_output)  # Classification binaire
    model = Model(inputs, outputs)
    return model

# Construire et compiler le modèle GRU
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
gru_model = build_gru_model(input_shape=input_shape)
gru_model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(name="AUC")])  # Métrique AUC

# Entraîner le modèle avec les poids de classe
print("Entraînement du modèle GRU avec équilibrage des classes et métrique AUC...")
history = gru_model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    class_weight=class_weights,
    verbose=1
)

# Évaluer le modèle
print("Évaluation du modèle GRU...")
evaluation = gru_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"AUC sur les données de test: {evaluation[1]:.4f}")

# Matrice de confusion
y_pred_proba = gru_model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("Calcul de la matrice de confusion...")
confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion:")
print(confusion_matrix.numpy())

# Afficher la matrice de confusion
def plot_confusion_matrix(cm, class_names):
    cm = cm.numpy()  # Convertir en tableau NumPy
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='Étiquette réelle',
           xlabel='Étiquette prédite')
    plt.title("Matrice de confusion")
    plt.show()

plot_confusion_matrix(confusion_matrix, class_names=["Classe 0", "Classe 1"])




#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Exemple de classe Dataset pour organiser les données
class GeographicDataset(Dataset):
    def __init__(self, data, target_col, sequence_col, group_col):
        """
        data: Pandas DataFrame contenant les données.
        target_col: Le nom de la colonne cible (classe à prédire).
        sequence_col: Liste des colonnes qui constituent la séquence d'entrée.
        group_col: Le nom de la colonne pour grouper les séries (par exemple, codgeo).
        """
        self.data = data
        self.target_col = target_col
        self.sequence_col = sequence_col
        self.group_col = group_col

        # Préparer les séquences par groupe géographique
        self.groups = self.data.groupby(self.group_col)
        self.sequences = []
        self.targets = []

        for _, group in self.groups:
            seq = group[sequence_col].values
            target = group[target_col].values[0]  # Supposons que la cible est la même pour tout le groupe
            self.sequences.append(seq)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)

# Définition du modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # On ne garde que le dernier état caché
        out = self.fc(h_n[-1])  # Utilisation de l'état final du dernier layer LSTM
        return out

# Exemple de préparation des données
# Remplacez ceci par votre propre chargement de données

sequence_cols = [ 'PRENEI_MENS', 'PRELIQ_MENS', 'PRETOTM_MENS', 'T_MENS',
       'EVAP_MENS', 'ETP_MENS', 'PE_MENS', 'SWI_MENS', 'DRAINC_MENS',
       'RUNC_MENS']

# Création du Dataset et DataLoader
dataset = GeographicDataset(data, target_col='dry', sequence_col=sequence_cols, group_col='codgeo')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialisation du modèle et des paramètres
input_size = len(sequence_cols)  # Nombre de caractéristiques dans chaque étape temporelle
hidden_size = 16
num_layers = 1
num_classes = len(data['dry'].unique())

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, targets in dataloader:
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

train_model(model, dataloader, criterion, optimizer, epochs=10)

# Prédiction sur de nouvelles données
def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in dataloader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
    return predictions

predictions = predict(model, dataloader)
print("Prédictions :", predictions)
#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Assume `data` is already loaded as a DataFrame
data = data2.drop(columns=['codgeo','reg', 'target','year_month_temp', 'DATE_temp', 'libgeo'])

# Step 1: Preprocessing
# Drop unsupported geometry column
if 'geometry' in data.columns:
    data = data.drop(columns=['geometry'])

# Convert `DATE_temp` to numeric features (year and month) and drop original column
if 'DATE_ville' in data.columns:
    data['month'] = data['DATE_ville'].dt.month
    data = data.drop(columns=['DATE_ville'])

# Convert object columns to category and encode as integers
for col in data.select_dtypes(include=['object']).columns:
    if col not in ['dry']:  # Exclude target column
        data[col] = data[col].astype('category')

for col in data.select_dtypes(include=['category']).columns:
    data[col] = data[col].cat.codes

# Ensure all columns are numeric
# Step 2: Define Features and Target
X = data.drop(columns=['dry'])  # Features
y = data['dry']                 # Target

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardization of Numeric Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Hyperparameter Grid Search
print("Starting hyperparameter grid search...")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', scale_pos_weight=len(y_train) / sum(y_train) - 1),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=3,
    n_jobs=-1
)

# Train Grid Search
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Step 6: Evaluation
# Predictions
print("Evaluating the best model...")
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # Predicted probabilities
y_pred = (y_pred_proba > 0.5).astype(int)  # Binary predictions based on threshold

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display Confusion Matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()

# Step 7: Save Results for Each Model in Grid
results = pd.DataFrame(grid_search.cv_results_)
results['mean_test_score'] = results['mean_test_score'].round(4)

# Iterate over each parameter set and store confusion matrices
conf_matrices = []
for params in results['params']:
    temp_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='auc')
    temp_model.fit(X_train_scaled, y_train)
    temp_y_pred = temp_model.predict(X_test_scaled)
    temp_conf_matrix = confusion_matrix(y_test, temp_y_pred)
    conf_matrices.append({
        'params': params,
        'conf_matrix': temp_conf_matrix,
        'normalized_conf_matrix': (temp_conf_matrix.astype('float') / temp_conf_matrix.sum(axis=1)[:, np.newaxis])
    })

# Save all confusion matrices with percentages
for idx, cm_info in enumerate(conf_matrices):
    print(f"Model {idx+1}: {cm_info['params']}")
    print("Confusion Matrix:")
    print(cm_info['conf_matrix'])
    print("Normalized Confusion Matrix (%):")
    print(cm_info['normalized_conf_matrix'] * 100)
