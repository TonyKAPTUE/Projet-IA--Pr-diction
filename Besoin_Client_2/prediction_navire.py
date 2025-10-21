import numpy as np
import joblib

# Chargement des objets préentraînés
scaler = joblib.load("scaler.save")
label_encoder = joblib.load("label_encoder.save")
model = joblib.load("random_forest_best_model.pkl")  

def predire_navire(length, width, draft):
    specs = np.array([[length, width, draft]])
    specs_scaled = scaler.transform(specs)
    prediction_encoded = model.predict(specs_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    return prediction_label[0]

if __name__ == "__main__":
    longueur = float(input(" Longueur (m) : "))
    largeur = float(input(" Largeur (m) : "))
    tirant_eau = float(input(" Tirant d'eau (m) : "))
    type_prédit = predire_navire(longueur, largeur, tirant_eau)
    print(f" Type de navire prédit : {type_prédit}")
