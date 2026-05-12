import pandas as pd
import numpy as np
from sklearn import linear_model

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
    
    print(f"\n LISTA DE EQUIPOS ({num_teams} detectados):")
    print("-" * 40)
    for i, name in enumerate(team_names):
        print(f" {i} -> {name}")
    print("-" * 40)
    
    X = df_x.values
    
    y_scores = []
    raw_y = df_y.values
    for row in raw_y:
        scores = [s for s in row if s > 0]
        if len(scores) < 2:
            y_scores.append([0, 0])
            continue
        midpoint = len(row) // 2
        score_home = sum(row[:midpoint])
        score_away = sum(row[midpoint:])
        y_scores.append([score_home, score_away])
    
    Y = np.array(y_scores)

    lr = linear_model.LinearRegression()
    lr.fit(X, Y)

    print("\n INGRESA LOS DATOS DEL PARTIDO:")
    try:
        idx_home = int(input(f"NÚMERO del Local: "))
        idx_away = int(input(f"NÚMERO del Visitante: "))
    except ValueError:
        print("Error: Debes escribir el número ID.")
        return

    name_home = team_names[idx_home]
    name_away = team_names[idx_away]

    input_vec = np.zeros((1, total_columns))
    input_vec[0, idx_home] = 1            
    input_vec[0, idx_away + num_teams] = 1 

    prediction = lr.predict(input_vec)
    pred_home = prediction[0][0]
    pred_away = prediction[0][1]

    # 7. RESULTADO
    print("\n" + "="*40)
    print(f"PREDICCIÓN DE MARCADOR")
    print(f"   {name_home} vs {name_away}")
    print("="*40)
    print(f"Local: {name_home}: {pred_home:.1f} puntos")
    print(f"Visitante: {name_away}:  {pred_away:.1f} puntos")
    print("-" * 40)
    
    margin = abs(pred_home - pred_away)
    if pred_home > pred_away:
        print(f"GANADOR: {name_home} (local) por {margin:.1f} puntos")
    else:
        print(f"GANADOR: {name_away} (visitante) por {margin:.1f} puntos")
    print("="*40)

if __name__ == '__main__':
    main()
