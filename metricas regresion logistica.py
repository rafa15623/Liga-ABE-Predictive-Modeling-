import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

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
    
    X_raw = df_x.values
    raw_y = df_y.values
    n_samples = X_raw.shape[0]
    
    team_stats = {i: {'wins': 0, 'games': 0} for i in range(num_teams)}
    
    X_train_list = []
    Y_train_list = []
    
    for i in range(n_samples):
        home_idx = np.argmax(X_raw[i, :num_teams])
        away_idx = np.argmax(X_raw[i, num_teams:])
        
        h_pct = (team_stats[home_idx]['wins'] + 0.5) / (team_stats[home_idx]['games'] + 1)
        a_pct = (team_stats[away_idx]['wins'] + 0.5) / (team_stats[away_idx]['games'] + 1)
        
        current_features = np.zeros(total_columns + 2)
        current_features[:total_columns] = X_raw[i]
        current_features[-2] = h_pct
        current_features[-1] = a_pct
        
        row_y = raw_y[i]
        scores = [s for s in row_y if s > 0]
        is_unplayed = len(scores) < 2
        
        if not is_unplayed:
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
            else: 
                team_stats[away_idx]['wins'] += 1

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    
    if len(X_train) == 0:
        print("No se encontraron partidos jugados para entrenar el modelo.")
        return
        
    print(f"Total de partidos utilizados para la evaluación: {len(X_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(
        X_train_scaled, Y_train, test_size=0.2, shuffle=False
    )
    
    clf = LogisticRegression(C=0.5, solver='liblinear', max_iter=1000)
    clf.fit(X_split_train, Y_split_train)
    
    predicciones = clf.predict(X_split_test)
    probabilidades = clf.predict_proba(X_split_test)[:, 1] 
    
    acc = accuracy_score(Y_split_test, predicciones)
    precision = precision_score(Y_split_test, predicciones, zero_division=0)
    recall = recall_score(Y_split_test, predicciones, zero_division=0)
    f1 = f1_score(Y_split_test, predicciones, zero_division=0)
    roc_auc = roc_auc_score(Y_split_test, probabilidades)
    cm = confusion_matrix(Y_split_test, predicciones)
    
    print("\n" + "="*60)
    print("        MÉTRICAS DE RENDIMIENTO (CONJUNTO DE PRUEBA)        ")
    print("="*60)
    print(f" Accuracy (Exactitud general): {acc:.2%}")
    print(f" Precision (Precisión):        {precision:.2%}")
    print(f" Recall (Sensibilidad):        {recall:.2%}")
    print(f" F1-Score:                     {f1:.2%}")
    print(f" ROC-AUC:                      {roc_auc:.2%}")
    print("-" * 60)
    print(" MATRIZ DE CONFUSIÓN:")
    print(f" [{cm[0][0]:3d}] Verdaderos Negativos  (Derrota Local predicha correctamente)")
    print(f" [{cm[0][1]:3d}] Falsos Positivos      (Predijo Victoria, pero fue Derrota)")
    print(f" [{cm[1][0]:3d}] Falsos Negativos      (Predijo Derrota, pero fue Victoria)")
    print(f" [{cm[1][1]:3d}] Verdaderos Positivos  (Victoria Local predicha correctamente)")
    print("="*60)
    
    coef_home_pct = clf.coef_[0][-2]
    coef_away_pct = clf.coef_[0][-1]
    
    print(f"\n Importancia asignada al % Victoria Local:  {coef_home_pct:.4f}")
    print(f" Importancia asignada al % Victoria Visita: {coef_away_pct:.4f}")

if __name__ == '__main__':
    main()
