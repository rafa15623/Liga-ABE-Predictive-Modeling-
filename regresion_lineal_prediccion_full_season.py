import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    team_stats = {i: {'wins': 0, 'games': 0, 'pts_for': 0, 'pts_against': 0} for i in range(num_teams)}
    
    reporte = {name: {
        'Equipo': name,
        'Victorias Reales': 0,
        'Derrotas Reales': 0,
        'Victorias Predichas': 0,
        'Derrotas Predichas': 0
    } for name in team_names}
    
    X_train_list = []
    Y_train_list = []
    partidos_pendientes_raw = []
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:])
        home_name = team_names[home_idx]
        away_name = team_names[away_idx]
        
        h_games = team_stats[home_idx]['games']
        a_games = team_stats[away_idx]['games']
        
        h_pct = (team_stats[home_idx]['wins'] + 0.5) / (h_games + 1)
        a_pct = (team_stats[away_idx]['wins'] + 0.5) / (a_games + 1)
        
        h_avg_f = team_stats[home_idx]['pts_for'] / h_games if h_games > 0 else 80.0
        h_avg_a = team_stats[home_idx]['pts_against'] / h_games if h_games > 0 else 80.0
        a_avg_f = team_stats[away_idx]['pts_for'] / a_games if a_games > 0 else 80.0
        a_avg_a = team_stats[away_idx]['pts_against'] / a_games if a_games > 0 else 80.0
        
        current_features = np.zeros(total_columns + 6)
        current_features[:total_columns] = X_raw[i]
        current_features[-6] = h_pct
        current_features[-5] = a_pct
        current_features[-4] = h_avg_f
        current_features[-3] = h_avg_a
        current_features[-2] = a_avg_f
        current_features[-1] = a_avg_a
        
        row_y = raw_y[i]
        scores = [s for s in row_y if s > 0]
        is_unplayed = len(scores) < 2
        
        if is_unplayed:
            partidos_pendientes_raw.append((i, home_idx, away_idx, home_name, away_name))
        else:
            midpoint = len(row_y) // 2
            score_home = sum(row_y[:midpoint])
            score_away = sum(row_y[midpoint:])
            winner_is_home = score_home > score_away
            
            X_train_list.append(current_features)
            Y_train_list.append([score_home, score_away])
            
            team_stats[home_idx]['games'] += 1
            team_stats[away_idx]['games'] += 1
            team_stats[home_idx]['pts_for'] += score_home
            team_stats[home_idx]['pts_against'] += score_away
            team_stats[away_idx]['pts_for'] += score_away
            team_stats[away_idx]['pts_against'] += score_home
            
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
    
    if len(X_train) == 0:
        print("No hay partidos jugados.")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(
        X_train_scaled, Y_train, test_size=0.2, random_state=42
    )
    
    lr_prueba = LinearRegression()
    lr_prueba.fit(X_split_train, Y_split_train)
    
    correct = 0
    homeDistance = 0
    awayDistance = 0
    
    for t in range(len(X_split_test)):
        pred = lr_prueba.predict([X_split_test[t]])[0]
        real = Y_split_test[t]
        
        homeDistance += abs(pred[0] - real[0])
        awayDistance += abs(pred[1] - real[1])
        
        if (pred[0] > pred[1]) == (real[0] > real[1]):
            correct += 1

    n_test = len(X_split_test)
    total_distance = homeDistance + awayDistance
    
    print("\n" + "="*50)
    print(" RENDIMIENTO MODELO LINEAL (CON PCT Y PROMEDIOS)")
    print("="*50)
    print(f"Error Promedio Total:  {total_distance/n_test:.2f} pts")
    print(f"Error Promedio Local:  {homeDistance/n_test:.2f} pts")
    print(f"Error Promedio Visita: {awayDistance/n_test:.2f} pts")
    print("-" * 50)
    print(f"PRECISIÓN (Ganador):   {correct/n_test:.2%}")
    print("="*50)

    lr = LinearRegression()
    lr.fit(X_train_scaled, Y_train)
    
    predicciones_detalle = []
    
    for game in partidos_pendientes_raw:
        i, home_idx, away_idx, home_name, away_name = game
        
        h_games = team_stats[home_idx]['games']
        a_games = team_stats[away_idx]['games']
        
        h_pct = (team_stats[home_idx]['wins'] + 0.5) / (h_games + 1)
        a_pct = (team_stats[away_idx]['wins'] + 0.5) / (a_games + 1)
        h_avg_f = team_stats[home_idx]['pts_for'] / h_games
        h_avg_a = team_stats[home_idx]['pts_against'] / h_games
        a_avg_f = team_stats[away_idx]['pts_for'] / a_games
        a_avg_a = team_stats[away_idx]['pts_against'] / a_games
        
        current_features = np.zeros(total_columns + 6)
        current_features[:total_columns] = X_raw[i]
        current_features[-6] = h_pct
        current_features[-5] = a_pct
        current_features[-4] = h_avg_f
        current_features[-3] = h_avg_a
        current_features[-2] = a_avg_f
        current_features[-1] = a_avg_a
        
        curr_scaled = scaler.transform([current_features])
        pred = lr.predict(curr_scaled)[0]
        pred_home = pred[0]
        pred_away = pred[1]
        
        winner_is_home = pred_home > pred_away
        
        # Actualización dinámica interna
        team_stats[home_idx]['games'] += 1
        team_stats[away_idx]['games'] += 1
        team_stats[home_idx]['pts_for'] += pred_home
        team_stats[home_idx]['pts_against'] += pred_away
        team_stats[away_idx]['pts_for'] += pred_away
        team_stats[away_idx]['pts_against'] += pred_home
        
        if winner_is_home:
            team_stats[home_idx]['wins'] += 1
            reporte[home_name]['Victorias Predichas'] += 1
            reporte[away_name]['Derrotas Predichas'] += 1
            ganador_nombre = home_name
        else:
            team_stats[away_idx]['wins'] += 1
            reporte[away_name]['Victorias Predichas'] += 1
            reporte[home_name]['Derrotas Predichas'] += 1
            ganador_nombre = away_name
            
        predicciones_detalle.append({
            'Equipo Local': home_name,
            'Puntos Local (Pred)': round(pred_home, 1),
            'Puntos Visita (Pred)': round(pred_away, 1),
            'Equipo Visitante': away_name,
            'Ganador': ganador_nombre,
            'Margen de Victoria': round(abs(pred_home - pred_away), 1)
        })

    df_reporte = pd.DataFrame(list(reporte.values()))
    
    df_reporte['Proy. Victorias'] = df_reporte['Victorias Reales'] + df_reporte['Victorias Predichas']
    df_reporte['Proy. Derrotas'] = df_reporte['Derrotas Reales'] + df_reporte['Derrotas Predichas']
    
    juegos_totales = df_reporte['Proy. Victorias'] + df_reporte['Proy. Derrotas']
    df_reporte['% Predictivo'] = np.where(juegos_totales > 0, df_reporte['Proy. Victorias'] / juegos_totales, 0)
    
    df_reporte = df_reporte.sort_values(
        by=['% Predictivo', 'Proy. Victorias', 'Victorias Reales'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    df_reporte['% Predictivo'] = df_reporte['% Predictivo'].apply(lambda x: f"{x:.3f}")
    
    columnas_finales = [
        'Equipo', 'Victorias Reales', 'Derrotas Reales', 
        'Victorias Predichas', 'Derrotas Predichas', 
        'Proy. Victorias', 'Proy. Derrotas', '% Predictivo'
    ]
    df_reporte = df_reporte[columnas_finales]
    
    print("\n" + "="*100)
    print(" " * 25 + "TABLA DE PROYECCIONES (REGRESIÓN LINEAL)")
    print("="*100)
    print(df_reporte.to_string(index=True))
    print("="*100)
    
    df_reporte.to_excel("Tabla_Proyecciones_Puntos.xlsx", index=False)
    
    if len(predicciones_detalle) > 0:
        pd.DataFrame(predicciones_detalle).to_excel("Calendario_Predicciones_Puntos.xlsx", index=False)
        print("\nArchivos 'Tabla_Proyecciones_Puntos.xlsx' y 'Calendario_Predicciones_Puntos.xlsx' guardados con éxito.")

if __name__ == '__main__':
    main()