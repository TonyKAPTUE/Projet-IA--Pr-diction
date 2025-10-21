# predict_trajectory_plot.py

import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt # On utilise matplotlib pour le graphique

def predict_ship_trajectory(ship_data):
    """Charge les modèles et prédit la trajectoire future."""
    try:
        model_columns = joblib.load('regression_model_columns.pkl')
        models = {}
        for horizon in [5, 10, 15]:
            models[f'lat_{horizon}'] = joblib.load(f'model_lat_{horizon}min.pkl')
            models[f'lon_{horizon}'] = joblib.load(f'model_lon_{horizon}min.pkl')
    except FileNotFoundError:
        print("ERREUR: Fichiers de modèle (.pkl) introuvables. Lancez d'abord le script d'entraînement.")
        return None, None
    
    # Préparation des données d'entrée
    df = pd.DataFrame([ship_data])
    df_encoded = pd.get_dummies(df)
    df_processed = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Prédiction
    predictions = {}
    for horizon in [5, 10, 15]:
        pred_lat = models[f'lat_{horizon}'].predict(df_processed)[0]
        pred_lon = models[f'lon_{horizon}'].predict(df_processed)[0]
        predictions[horizon] = {'LAT': pred_lat, 'LON': pred_lon}
    return predictions

# --- NOUVELLE FONCTION DE VISUALISATION SIMPLE ---
def visualiser_trajectoire_2d(start_point, predictions):
    """
    Crée un graphique 2D simple (LAT vs LON) et le sauvegarde en image.
    """
    print("\nCréation du graphique de la trajectoire prédite...")

    # On prépare les listes de coordonnées pour le graphique
    all_lats = [start_point['LAT']]
    all_lons = [start_point['LON']]
    
    predicted_lats = []
    predicted_lons = []
    
    for horizon in sorted(predictions.keys()):
        pos = predictions[horizon]
        all_lats.append(pos['LAT'])
        all_lons.append(pos['LON'])
        predicted_lats.append(pos['LAT'])
        predicted_lons.append(pos['LON'])
        
    # Création du graphique
    plt.figure(figsize=(10, 8))
    
    # 1. Dessiner la trajectoire en ligne pointillée
    plt.plot(all_lons, all_lats, color='gray', linestyle='--', marker='o', zorder=1)
    
    # 2. Dessiner le point de départ en bleu
    plt.scatter(start_point['LON'], start_point['LAT'], 
                color='blue', label='Point de départ (T+0)', s=100, zorder=5)
    
    # 3. Dessiner les points prédits en rouge
    plt.scatter(predicted_lons, predicted_lats, 
                color='red', label='Prédictions', marker='x', s=100)
    
    # Ajouter des étiquettes de temps à côté des points prédits
    for i, horizon in enumerate(sorted(predictions.keys())):
        plt.text(predicted_lons[i] + 0.001, predicted_lats[i] + 0.001, f'T+{horizon} min')

    # Mise en forme du graphique
    plt.title("Prédiction de Trajectoire")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Assure que les échelles sont équivalentes
    
    # Sauvegarde de l'image
    filename = "prediction_trajectoire_plot.png"
    plt.savefig(filename)
    print(f"Graphique de la trajectoire sauvegardé sous '{filename}'")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prédire la trajectoire future d'un navire.")
    parser.add_argument('--sog', type=float, required=True, help="Vitesse (SOG)")
    parser.add_argument('--cog', type=float, required=True, help="Cap (COG)")
    parser.add_argument('--heading', type=float, required=True, help="Orientation (Heading)")
    parser.add_argument('--length', type=float, required=True, help="Longueur")
    parser.add_argument('--width', type=float, required=True, help="Largeur")
    parser.add_argument('--vesseltype', type=int, required=True, help="Type de navire (ex: 70)")
    parser.add_argument('--start_lat', type=float, required=True, help="Latitude de départ")
    parser.add_argument('--start_lon', type=float, required=True, help="Longitude de départ")

    args = parser.parse_args()
    
    start_point_data = {'LAT': args.start_lat, 'LON': args.start_lon}
    new_ship_data = {
        'SOG': args.sog, 'COG': args.cog, 'Heading': args.heading,
        'Length': args.length, 'Width': args.width, f'VesselType_{args.vesseltype}.0': 1
    }
    
    # Lancement de la prédiction
    result = predict_ship_trajectory(new_ship_data)
    
    if result:
        future_positions = result
        print("\n--- Prédictions de Trajectoire ---")
        for horizon, position in future_positions.items():
            print(f"Position estimée à T + {horizon} minutes : Latitude={position['LAT']:.4f}, Longitude={position['LON']:.4f}")
        
        # On appelle la nouvelle fonction de visualisation 2D
        visualiser_trajectoire_2d(start_point_data, future_positions)