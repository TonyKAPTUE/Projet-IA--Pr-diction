# train_trajectory_model.py (Version finale et propre)

import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def preparer_donnees_multi_horizon(df, horizons):
    print(f"\n--- Préparation des données pour les horizons T + {horizons} minutes ---")
    df_prepared = df.copy()
    for horizon in horizons:
        df_temp = df.copy()
        df_temp['join_time'] = df_temp['BaseDateTime'] - pd.Timedelta(minutes=horizon)
        df_future = df_temp[['MMSI', 'join_time', 'LAT', 'LON']].rename(
            columns={'LAT': f'LAT_plus_{horizon}min', 'LON': f'LON_plus_{horizon}min'}
        )
        df_prepared = pd.merge_asof(
            left=df_prepared.sort_values('BaseDateTime'), right=df_future.sort_values('join_time'),
            left_on='BaseDateTime', right_on='join_time', by='MMSI', direction='forward', tolerance=pd.Timedelta(minutes=2)
        )
    target_columns = [f'{prefix}_plus_{h}min' for h in horizons for prefix in ['LAT', 'LON']]
    df_prepared.dropna(subset=target_columns, inplace=True)
    return df_prepared

def entrainer_et_sauvegarder(df_prepared, horizons):
    print("\n--- Division des données en entraînement et test ---")
    train_df, test_df = train_test_split(df_prepared, test_size=0.2, random_state=42)

    features = ['SOG', 'COG', 'Heading', 'Length', 'Width'] + [col for col in train_df.columns if 'VesselType_' in col]
    print(f"\nFeatures utilisées pour l'entraînement : {features}")
    
    joblib.dump(features, 'regression_model_columns.pkl')
    print("Noms des colonnes du modèle sauvegardés dans 'regression_model_columns.pkl'")

    X_train = train_df[features]
    
    for horizon in horizons:
        print(f"\n--- Entraînement pour l'horizon T + {horizon} minutes ---")
        target_lat, target_lon = f'LAT_plus_{horizon}min', f'LON_plus_{horizon}min'
        y_train_lat, y_train_lon = train_df[target_lat], train_df[target_lon]

        model_lat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10).fit(X_train, y_train_lat)
        joblib.dump(model_lat, f'model_lat_{horizon}min.pkl')
        print(f"Modèle pour LAT +{horizon}min sauvegardé.")
        
        model_lon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10).fit(X_train, y_train_lon)
        joblib.dump(model_lon, f'model_lon_{horizon}min.pkl')
        print(f"Modèle pour LON +{horizon}min sauvegardé.")
    
    print("\nEntraînement et sauvegarde des 6 modèles terminés.")

if __name__ == '__main__':
    try:
        print("--- Démarrage du script d'entraînement pour le Besoin Client 3 ---")
        df_main = pd.read_csv('export_IA.csv', sep=None, engine='python')
        print("Fichier CSV chargé avec succès.")
        
        df_main.columns = df_main.columns.str.replace('"', '', regex=False).str.strip()
        df_main['BaseDateTime'] = pd.to_datetime(df_main['BaseDateTime'])
        df_main.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
        for col in ['Heading', 'Length', 'Width']:
            if col in df_main.columns:
                df_main[col] = df_main[col].fillna(df_main[col].median())
        
        if 'VesselType' in df_main.columns:
            df_main = pd.get_dummies(df_main, columns=['VesselType'], prefix='VesselType', dummy_na=True)
            
    except Exception as e:
        print(f"ERREUR lors du chargement ou pré-traitement initial : {e}")
    else:
        horizons_a_predire = [5, 10, 15]
        df_prepared = preparer_donnees_multi_horizon(df_main, horizons_a_predire)
        
        if not df_prepared.empty:
            entrainer_et_sauvegarder(df_prepared, horizons_a_predire)
        else:
            print("Aucune donnée exploitable après la préparation des horizons.")
            
        print("\nScript d'entraînement terminé.")