# Liga-ABE-Predictive-Modeling

Predictive models for the Liga ABE (Asociación de Básquetbol Estudiantil) using machine learning techniques to forecast game outcomes.

## Project Structure

### Data Acquisition & Preprocessing
* `Web_scrapping.ipynb`: Extracts data from the official Liga ABE website and performs data cleaning and preparation.

### Modeling & Training
* `Modelo regresion logistica.py` & `Modelo regresion lineal.py`: Core scripts for training the initial models focusing on the first four weeks of the season.
* `metricas regresion logistica.py` & `metricas regresion lineal.py`: Evaluation scripts to calculate performance metrics and validate model accuracy.

### Prediction & Results
* `Predictor regresion logistica.py` & `Predictor regresion lineal.py`: Interactive tools to generate game predictions. 
    * *Note: Ensure the corresponding model script is run before executing the predictor.*
* `regresion_logistica_record_estatico.py` & `regresion_lineal_prediccion_full_season.py`: Advanced models designed to predict remaining games and generate summary documents with the projected results.

### Data Files
The project uses One-Hot Encoded datasets organized by timeframe:
* **Weekly Data:** `Partidos_OneHot_Binario` and `Resultados_OneHot_38cols` (Versions 1 through 4) contain training and testing data for the opening weeks of prediction.
* **Full Season Data:** `Partidos_OneHot_Binario_full_season` and `Resultados_OneHot_full_season` contain the dataset for the remainder of the competitive season.
