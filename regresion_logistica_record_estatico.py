import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN DE ARCHIVOS ---
FILE_X = "Partidos_OneHot_Binario_full_season.csv"
FILE_Y = "Resultados_OneHot_full_season.csv" 

def main():
    try:
        df_x = pd.read_csv(FILE_X, header=0)
        df_y = pd.read_csv(FILE_Y, header=0)
    except FileNotFoundError:
        print("Error: No encuentro los archivos CSV.")
        return

    total_columns = df_x.shape[1]
    num_teams = total_columns // 2
    team_names = list(df_x.columns[:num_teams])
    
    X_raw = df_x.values
    raw_y = df_y.values
    
    n_samples = X_raw.shape[0]
    
    team_stats = {i: {'wins': 0, 'games': 0} for i in range(num_teams)}
    reporte = {name: {
        'Equipo': name,
        'Victorias Reales': 0,
        'Derrotas Reales': 0,
        'Victorias Predichas': 0,
        'Derrotas Predichas': 0
    } for name in team_names}
    
    X_train_list = []
    Y_train_list = []
    X_predict_list = []
    predict_info = [] 
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:])
        home_name = team_names[home_idx]
        away_name = team_names[away_idx]
        
        h_pct = (team_stats[home_idx]['wins'] + 0.5) / (team_stats[home_idx]['games'] + 1)
        a_pct = (team_stats[away_idx]['wins'] + 0.5) / (team_stats[away_idx]['games'] + 1)
        
        current_features = np.zeros(total_columns + 2)
        current_features[:total_columns] = X_raw[i]
        current_features[-2] = h_pct
        current_features[-1] = a_pct
        
        row_y = raw_y[i]
        scores = [s for s in row_y if s > 0]
        is_unplayed = len(scores) < 2
        
        if is_unplayed:
            X_predict_list.append(current_features)
            predict_info.append({'home': home_name, 'away': away_name})
        else:
            midpoint = len(row_y) // 2
            score_home = sum(row_y[:midpoint])
            score_away = sum(row_y[midpoint:])
            
            winner_is_home = score_home > score_away
            
            Y_train_list.append(1 if winner_is_home else 0)
            X_train_list.append(current_features)
            
            team_stats[home_idx]['games'] += 1
            team_stats[away_idx]['games'] += 1
            
            if winner_is_home: 
                team_stats[home_idx]['wins'] += 1
                reporte[home_name]['Victorias Reales'] += 1
                reporte[away_name]['Derrotas Reales'] += 1
            else: 
                team_stats[away_idx]['wins'] += 1
                reporte[away_name]['Victorias Reales'] += 1
                reporte[home_name]['Derrotas Reales'] += 1

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    X_predict = np.array(X_predict_list)
    
    if len(X_train) == 0:
        print("No se encontraron partidos jugados para entrenar el modelo.")
        return

    # --- ENTRENAMIENTO Y EVALUACIÓN ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(
        X_train_scaled, Y_train, test_size=0.2, random_state=42
    )
    
    clf_prueba = LogisticRegression(C=0.5, solver='liblinear', max_iter=1000)
    clf_prueba.fit(X_split_train, Y_split_train)
    
    predicciones_prueba = clf_prueba.predict(X_split_test)
    acc = accuracy_score(Y_split_test, predicciones_prueba)
    
    print("\n" + "="*50)
    print(f"RENDIMIENTO DEL MODELO (ACCURACY): {acc:.2%}")
    print("="*50)
    
    clf = LogisticRegression(C=0.5, solver='liblinear', max_iter=1000)
    clf.fit(X_train_scaled, Y_train)
    
    coef_home_pct = clf.coef_[0][-2]
    coef_away_pct = clf.coef_[0][-1]
    
    print(f"Peso del % Victoria Local: {coef_home_pct:.4f}")
    print(f"Peso del % Victoria Visita: {coef_away_pct:.4f}")
    print("="*50)
    
    predicciones_detalle = []  
    
    if len(X_predict) > 0:
        X_predict_scaled = scaler.transform(X_predict)
        probs = clf.predict_proba(X_predict_scaled)
        
        for j in range(len(X_predict)):
            info = predict_info[j]
            home_name = info['home']
            away_name = info['away']
            
            prob_visita = probs[j][0]
            prob_local = probs[j][1]
            
            if prob_local > prob_visita:
                reporte[home_name]['Victorias Predichas'] += 1
                reporte[away_name]['Derrotas Predichas'] += 1
                ganador_nombre = home_name
            else:
                reporte[away_name]['Victorias Predichas'] += 1
                reporte[home_name]['Derrotas Predichas'] += 1
                ganador_nombre = away_name
                
            predicciones_detalle.append({
                'Equipo Local': home_name,
                'Prob. Victoria Local (%)': round(prob_local * 100, 2),
                'Prob. Victoria Visita (%)': round(prob_visita * 100, 2),
                'Equipo Visitante': away_name,
                'Ganador': ganador_nombre
            })

    df_reporte = pd.DataFrame(list(reporte.values()))
    
    df_reporte['Proy. Victorias'] = df_reporte['Victorias Reales'] + df_reporte['Victorias Predichas']
    df_reporte['Proy. Derrotas'] = df_reporte['Derrotas Reales'] + df_reporte['Derrotas Predichas']
    
    juegos_totales = df_reporte['Proy. Victorias'] + df_reporte['Proy. Derrotas']
    df_reporte['% Predictivo'] = np.where(
        juegos_totales > 0, 
        df_reporte['Proy. Victorias'] / juegos_totales, 
        0
    )
    
    df_reporte = df_reporte.sort_values(
        by=['% Predictivo', 'Proy. Victorias', 'Victorias Reales'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    df_reporte['% Predictivo'] = df_reporte['% Predictivo'].apply(lambda x: f"{x:.3f}")
    
    # Imprimir en consola (opcional)
    print("\n" + "="*95)
    print(" " * 25 + "TABLA DE PROYECCIONES DE LA TEMPORADA")
    print("="*95)
    print(df_reporte.to_string(index=True))
    print("="*95)
    
    # Archivo de la tabla de posiciones
    archivo_posiciones = "Tabla_Proyecciones.xlsx"
    df_reporte.to_excel(archivo_posiciones, index=False)
    print(f"\n Tabla general guardada en: {archivo_posiciones}")
    
    # Archivo del calendario de predicciones (partido a partido)
    if len(predicciones_detalle) > 0:
        df_predicciones = pd.DataFrame(predicciones_detalle)
        archivo_calendario = "Calendario_Predicciones.xlsx"
        df_predicciones.to_excel(archivo_calendario, index=False)
        print(f"n\ Detalle cronológico guardado en: {archivo_calendario}")
    else:
        print("\nNo hubo partidos nuevos para predecir.")

if __name__ == '__main__':
    main()