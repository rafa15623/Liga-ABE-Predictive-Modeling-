import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# ARCHIVOS
FILE_X = "Partidos_OneHot_Binario_(4).csv"
FILE_Y = "Resultados_OneHot_38cols_(4).csv" 

def no_shuffle_train_test_split(X, Y, percentage):
    indexFromPercentage = int(X.shape[0] * percentage)
    train_X, test_X = np.split(X, [indexFromPercentage])
    train_Y, test_Y = np.split(Y, [indexFromPercentage])
    return train_X, test_X, train_Y, test_Y

def load_data_robust(filename_x, filename_y):
    try:
        df_x = pd.read_csv(filename_x, header=0)
        df_y = pd.read_csv(filename_y, header=0)
    except FileNotFoundError:
        print("Error: No se encuentran los archivos CSV.")
        return None, None

    X = df_x.values
    
    y_scores = [] 
    
    raw_y = df_y.values
    for i in range(len(raw_y)):
        row = raw_y[i]
        scores = [s for s in row if s > 0]
        
        if len(scores) < 2:
            y_scores.append([0, 0]) 
            continue
            
        midpoint = len(row) // 2
        
        score_home = sum(row[:midpoint])
        score_away = sum(row[midpoint:])
        
        y_scores.append([score_home, score_away])

    return X, np.array(y_scores)

def testResults(lr, X_test, Y_test):
    correct = 0
    distance = 0
    homeDistance = 0
    awayDistance = 0
    
    for t in range(len(X_test)):
        prediction = lr.predict([X_test[t]])
        pred_home = prediction[0][0]
        pred_away = prediction[0][1]
        
        real_home = Y_test[t][0]
        real_away = Y_test[t][1]
        
        real_winner_is_home = real_home > real_away
        pred_winner_is_home = pred_home > pred_away
        
        if real_winner_is_home == pred_winner_is_home:
            correct += 1

        distance += abs(pred_home - real_home) + abs(pred_away - real_away)
        homeDistance += abs(pred_home - real_home)
        awayDistance += abs(pred_away - real_away)
    
    if X_test.shape[0] > 0:
        print("\n" + "="*30)
        print("RESULTADOS")
        print("="*30)
        print("Error Promedio Total: %f puntos" % (distance/float(X_test.shape[0])))
        print("Error Puntos Local:   %f" % (homeDistance/float(X_test.shape[0])))
        print("Error Puntos Visita:  %f" % (awayDistance/float(X_test.shape[0])))
        print("-" * 30)
        print("PRECISIÓN (Ganador):  %f" % (correct/float(len(X_test))))
        print("="*30)
    else:
        print("No test data found.")
    return

def dealWithLR():
    print("Entrenando Regresión Lineal (Predicción de Puntos)")
    
    X, Y = load_data_robust(FILE_X, FILE_Y)
    if X is None: return

    X_train, X_test, Y_train, Y_test = no_shuffle_train_test_split(X, Y, 0.9)
    print(f"Entrenando con {len(X_train)} datos. Probando con {len(X_test)} datos.")
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, Y_train)
    
    testResults(lr, X_test, Y_test) 

if __name__ == '__main__':
    dealWithLR()