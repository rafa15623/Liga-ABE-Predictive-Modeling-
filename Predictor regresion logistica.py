import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- ARCHIVOS ---
FILE_X = "Partidos_OneHot_Binario_(4).csv"
FILE_Y = "Resultados_OneHot_38cols_(4).csv" 

def main():
    try:
        df_x = pd.read_csv(FILE_X, header=0)
        df_y = pd.read_csv(FILE_Y, header=0)
    except FileNotFoundError:
        print("Error: No encuentro los CSV.")
        return

    total_columns = df_x.shape[1]
    num_teams = total_columns // 2
    team_names = list(df_x.columns[:num_teams])
    
    print(f"\n EQUIPOS:")
    print("-" * 40)
    for i, name in enumerate(team_names):
        print(f" {i} -> {name}")
    print("-" * 40)
    
    X_raw = df_x.values
    raw_y = df_y.values
    
    y_binary = []
    for row in raw_y:
        scores = [s for s in row if s > 0]
        if len(scores) < 2: 
            y_binary.append(0)
            continue
        midpoint = len(row) // 2
        if sum(row[:midpoint]) > sum(row[midpoint:]):
            y_binary.append(1) 
        else:
            y_binary.append(0) 
    y_binary = np.array(y_binary)

    n_samples = X_raw.shape[0]
    features_stats = np.zeros((n_samples, 2))
    team_stats = {i: {'wins': 0, 'games': 0} for i in range(num_teams)}
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:])
        
        h_pct = (team_stats[home_idx]['wins'] + 0.5) / (team_stats[home_idx]['games'] + 1)
        a_pct = (team_stats[away_idx]['wins'] + 0.5) / (team_stats[away_idx]['games'] + 1)
        features_stats[i, 0] = h_pct
        features_stats[i, 1] = a_pct
        
        winner_is_home = y_binary[i] == 1
        team_stats[home_idx]['games'] += 1
        team_stats[away_idx]['games'] += 1
        if winner_is_home: team_stats[home_idx]['wins'] += 1
        else: team_stats[away_idx]['wins'] += 1

    X_final = np.hstack((X_raw, features_stats))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    clf = LogisticRegression(C=0.5, solver='liblinear', max_iter=1000)
    clf.fit(X_scaled, y_binary)

    try:
        idx_home = int(input(f"Escribe el NÚMERO del Local: "))
        idx_away = int(input(f"Escribe el NÚMERO del Visitante: "))
    except ValueError:
        print("Error: Debes escribir el número ID (ej. 3).")
        return

    name_home = team_names[idx_home]
    name_away = team_names[idx_away]

    curr_h_wins = team_stats[idx_home]['wins']
    curr_h_games = team_stats[idx_home]['games']
    curr_h_pct = (curr_h_wins + 0.5) / (curr_h_games + 1)

    curr_a_wins = team_stats[idx_away]['wins']
    curr_a_games = team_stats[idx_away]['games']
    curr_a_pct = (curr_a_wins + 0.5) / (curr_a_games + 1)

    print(f"\n ANÁLISIS PREVIO:")
    print(f"Local: {name_home}: {curr_h_wins} victorias en {curr_h_games} juegos ({curr_h_pct:.1%})")
    print(f"Visitante: {name_away}: {curr_a_wins} victorias en {curr_a_games} juegos ({curr_a_pct:.1%})")

    input_vec = np.zeros((1, X_raw.shape[1] + 2)) 
    input_vec[0, idx_home] = 1          
    input_vec[0, idx_away + num_teams] = 1 
    input_vec[0, -2] = curr_h_pct       
    input_vec[0, -1] = curr_a_pct

    input_vec_scaled = scaler.transform(input_vec)

    prob = clf.predict_proba(input_vec_scaled)[0]
    prob_local = prob[1] * 100
    prob_visita = prob[0] * 100

    print("\n" + "="*40)
    print(f"PREDICCIÓN DEL MODELO")
    print("="*40)
    if prob_local > prob_visita:
        print(f"GANADOR: {name_home} (LOCAL)")
        print(f"Probabilidad: {prob_local:.2f}%")
    else:
        print(f"GANADOR: {name_away} (VISITANTE)")
        print(f"Probabilidad: {prob_visita:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()