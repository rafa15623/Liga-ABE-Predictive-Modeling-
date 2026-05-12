import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
FILE_X = "Partidos_OneHot_Binario_(4).csv"
FILE_Y = "Resultados_OneHot_38cols_(4).csv" 

def load_data_and_enrich(filename_x, filename_y):
    try:
        df_x = pd.read_csv(filename_x, header=0)
        df_y = pd.read_csv(filename_y, header=0)
    except FileNotFoundError:
        print("Error: No se encuentran los archivos CSV.")
        return None, None

    X_raw = df_x.values
    raw_y = df_y.values
    
    # Vector Y (Ganador: 1 Local, 0 Visita)
    y_binary = []
    for i in range(len(raw_y)):
        row = raw_y[i]
        scores = [s for s in row if s > 0]
        if len(scores) < 2:
            y_binary.append(0) 
            continue
        
        midpoint = len(row) // 2
        score_home = sum(row[:midpoint])
        score_away = sum(row[midpoint:])
        
        if score_home > score_away:
            y_binary.append(1)
        else:
            y_binary.append(0)
            
    y_binary = np.array(y_binary)
    
    n_samples = X_raw.shape[0]
    num_teams = X_raw.shape[1] // 2 
    
    features_stats = np.zeros((n_samples, 2))
    
    team_stats = {i: {'wins': 0, 'games': 0} for i in range(num_teams)}
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:]) 
        
        h_wins = team_stats[home_idx]['wins']
        h_games = team_stats[home_idx]['games']
        h_pct = (h_wins + 0.5) / (h_games + 1) 
        
        a_wins = team_stats[away_idx]['wins']
        a_games = team_stats[away_idx]['games']
        a_pct = (a_wins + 0.5) / (a_games + 1) 
        
        features_stats[i, 0] = h_pct
        features_stats[i, 1] = a_pct
        
        winner_is_home = y_binary[i] == 1
        
        team_stats[home_idx]['games'] += 1
        team_stats[away_idx]['games'] += 1
        
        if winner_is_home:
            team_stats[home_idx]['wins'] += 1
        else:
            team_stats[away_idx]['wins'] += 1

    X_final = np.hstack((X_raw, features_stats))
    
    return X_final, y_binary

def main():
    print("Entrenando Regresión Logística con porcentaje de victorias")
    
    X, Y = load_data_and_enrich(FILE_X, FILE_Y)
    
    if X is None: return

    split_idx = int(X.shape[0] * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Entrenando con {len(X_train)} juegos. Probando con {len(X_test)}.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=0.5, solver='liblinear', max_iter=1000)
    clf.fit(X_train, Y_train)
    
    predictions = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    
    acc = accuracy_score(Y_test, predictions)
    print("\n" + "="*30)
    print(f"NUEVA PRECISIÓN (accuracy): {acc:.2%}")
    print("="*30)
    
    coef_home_pct = clf.coef_[0][-2]
    coef_away_pct = clf.coef_[0][-1]
    
    print(f"Importancia del % Victoria Local: {coef_home_pct:.4f}")
    print(f"Importancia del % Victoria Visita: {coef_away_pct:.4f}")

if __name__ == '__main__':
    main()