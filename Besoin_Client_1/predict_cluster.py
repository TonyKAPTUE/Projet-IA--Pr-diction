import joblib
import numpy as np

def main():
    # Chargement des objets sauvegardés
    try:
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print(" Modèle ou scaler introuvable. Vérifie les noms de fichiers.")
        return

    # Saisie utilisateur
    print("Saisis les caractéristiques du navire :")
    try:
        lat = float(input("Latitude : "))
        lon = float(input("Longitude : "))
        sog = float(input("Speed Over Ground (SOG) : "))
        cog = float(input("Course Over Ground (COG) : "))
    except ValueError:
        print(" Entrée invalide. Veuillez entrer des nombres.")
        return

    # Prédiction
    features = np.array([[lat, lon, sog, cog]])
    features_scaled = scaler.transform(features)
    cluster = kmeans.predict(features_scaled)

    print(f"\n Le navire appartient au cluster : {cluster[0]}")

if __name__ == "__main__":
    main()