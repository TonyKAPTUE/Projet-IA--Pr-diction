# Projet_IA_Groupe_8
# Prédiction du Type de Navire

## Objectif

Ce script permet de prédire le type d’un navire en fonction de ses caractéristiques physiques : longueur, largeur et tirant d’eau. Il s'appuie sur un modèle Random Forest préalablement entraîné et enregistré.

## Fichiers inclus

- `prediction_navire.py` : script principal de prédiction via la console.
- `random_forest_best_model.pkl` : modèle Random Forest entraîné.
- `scaler.save` : normaliseur `StandardScaler`.
- `label_encoder.save` : encodeur d’étiquettes pour interpréter la sortie du modèle.

## Technologies utilisées

- Python 3
- NumPy
- Scikit-learn
- joblib

## Utilisation

1. Vérifiez que tous les fichiers requis sont dans le même dossier.
2. Ouvrez un terminal et exécutez :

```bash
python prediction_navire.py
