import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, 
    mean_absolute_error, mean_squared_error, r2_score
)

FILE_X = "Partidos_OneHot_Binario_full_season.csv"
FILE_Y = "Resultados_OneHot_full_season.csv" 

def main():
    try:
        df_x = pd.read_csv(FILE_X, header=0)
        df_y = pd.read_csv(FILE_Y, header=0)
    except FileNotFoundError:
        print("Error: No se encuentran los archivos CSV.")
        return

    total_columns = df_x.shape[1]
    num_teams = total_columns // 2
    team_names = list(df_x.columns[:num_teams])
    
    X_raw = df_x.values
    raw_y = df_y.values
    n_samples = X_raw.shape[0]
    
    team_stats = {i: {'wins': 0, 'games': 0, 'pts_for': 0, 'pts_against': 0} for i in range(num_teams)}
    
    X_train_list = []
    Y_train_list = []
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:])
        
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
        
        if not is_unplayed:
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
            else:
                team_stats[away_idx]['wins'] += 1

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    
    if len(X_train) == 0:
        print("No hay partidos jugados.")
        return

    print(f"Total de partidos utilizados para la evaluación: {len(X_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(
        X_train_scaled, Y_train, test_size=0.2, shuffle=False
    )
    
    lr = LinearRegression()
    lr.fit(X_split_train, Y_split_train)
    
    predicciones_puntos = lr.predict(X_split_test)
    
    Y_true_bin = (Y_split_test[:, 0] > Y_split_test[:, 1]).astype(int)
    Y_pred_bin = (predicciones_puntos[:, 0] > predicciones_puntos[:, 1]).astype(int)
    
    diferencia_puntos_pred = predicciones_puntos[:, 0] - predicciones_puntos[:, 1]
    
    y_true_flat = Y_split_test.flatten()
    y_pred_flat = predicciones_puntos.flatten()
    
    me_total = np.mean(y_true_flat - y_pred_flat)
    mae_total = mean_absolute_error(y_true_flat, y_pred_flat)
    mse_total = mean_squared_error(y_true_flat, y_pred_flat)
    rmse_total = np.sqrt(mse_total)
    
    r2_val = r2_score(y_true_flat, y_pred_flat)
    
    n_observaciones = len(y_true_flat)
    p_predictores = X_split_test.shape[1]
    
    if n_observaciones > p_predictores + 1:
        adj_r2_val = 1 - ((1 - r2_val) * (n_observaciones - 1) / (n_observaciones - p_predictores - 1))
    else:
        adj_r2_val = float('nan') 
    
    acc = accuracy_score(Y_true_bin, Y_pred_bin)
    precision = precision_score(Y_true_bin, Y_pred_bin, zero_division=0)
    recall = recall_score(Y_true_bin, Y_pred_bin, zero_division=0)
    f1 = f1_score(Y_true_bin, Y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(Y_true_bin, diferencia_puntos_pred)
    cm = confusion_matrix(Y_true_bin, Y_pred_bin)
    
    print("\n" + "="*60)
    print("        MÉTRICAS DE RENDIMIENTO (CONJUNTO DE PRUEBA)        ")
    print("="*60)
    print(" 1. MÉTRICAS DE PUNTOS GLOBALES (REGRESIÓN):")
    print(f" Mean Error (ME):                 {me_total:.2f} pts")
    print(f" Mean Absolute Error (MAE):       {mae_total:.2f} pts")
    print(f" Mean Squared Error (MSE):        {mse_total:.2f} pts²")
    print(f" Root Mean Sq. Error (RMSE):      {rmse_total:.2f} pts")
    print(f" R-cuadrado (R²):                 {r2_val:.4f}")
    print(f" R-cuadrado Ajustado (Adj R²):    {adj_r2_val:.4f}")
    print("-" * 60)
    print(" 2. MÉTRICAS DE CLASIFICACIÓN (GANADOR DEL PARTIDO):")
    print(f" Accuracy (Exactitud general): {acc:.2%}")
    print(f" Precision (Precisión):        {precision:.2%}")
    print(f" Recall (Sensibilidad):        {recall:.2%}")
    print(f" F1-Score:                     {f1:.2%}")
    print(f" ROC-AUC (vía margen puntos):  {roc_auc:.2%}")
    print("-" * 60)
    print(" MATRIZ DE CONFUSIÓN:")
    print(f" [{cm[0][0]:3d}] Verdaderos Negativos  (Derrota Local predicha correctamente)")
    print(f" [{cm[0][1]:3d}] Falsos Positivos      (Predijo Victoria, pero fue Derrota)")
    print(f" [{cm[1][0]:3d}] Falsos Negativos      (Predijo Derrota, pero fue Victoria)")
    print(f" [{cm[1][1]:3d}] Verdaderos Positivos  (Victoria Local predicha correctamente)")
    print("="*60)
    
    coefs_home = lr.coef_[0]
    
    print("\n IMPORTANCIA DE VARIABLES (Impacto en Puntos del Local):")
    print(f" Peso del % Victoria Local:        {coefs_home[-6]:.4f}")
    print(f" Peso del % Victoria Visita:       {coefs_home[-5]:.4f}")
    print(f" Peso Puntos Anotados (Local):     {coefs_home[-4]:.4f}")
    print(f" Peso Puntos Recibidos (Local):    {coefs_home[-3]:.4f}")
    print(f" Peso Puntos Anotados (Visita):    {coefs_home[-2]:.4f}")
    print(f" Peso Puntos Recibidos (Visita):   {coefs_home[-1]:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
