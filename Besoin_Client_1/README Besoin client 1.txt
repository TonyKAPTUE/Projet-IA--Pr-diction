# Projet_IA_Groupe_8
# Prédiction de Cluster pour un Navire

## Description

Ce script permet de prédire à quel **cluster** appartient un navire selon ses caractéristiques de navigation :
- Latitude
- Longitude
- Vitesse sur le fond (SOG)
- Cap sur le fond (COG)

Le modèle a été entraîné avec l’algorithme **KMeans**, et les objets nécessaires (modèle et scaler) sont chargés via la bibliothèque `joblib`.

## Fichiers requis

- `kmeans_model.pkl` : modèle KMeans entraîné
- `scaler.pkl` : objet StandardScaler correspondant aux données
- `predict_cluster.py` : script principal de prédiction

## Dépendances

- Python 3
- NumPy
- joblib

## Instructions d'utilisation

1. Vérifiez que tous les fichiers mentionnés ci-dessus sont dans le même dossier.
2. Ouvrez un terminal et exécutez :

```bash
python predict_cluster.py
