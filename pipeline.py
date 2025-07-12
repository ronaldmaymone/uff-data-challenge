import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# ----------------------------
# 1. Carregar dados (exemplo)
# ----------------------------

# Substitua por seu arquivo real
# df = pd.read_json('seu_arquivo.json')
# Aqui, dados fictícios:

np.random.seed(42)
n_samples = 500
df = pd.DataFrame({
    'id': np.arange(n_samples),
    'rate': np.random.normal(1000, 100, size=n_samples),
    'elapsed': np.random.uniform(5, 10, size=n_samples),
    'request_ticks': np.arange(n_samples)
})

# ----------------------------
# 2. Criar features de lag
# ----------------------------
# Não acho que precisa criar o lag pq já tem uma série de 10 valores gerados.
# def create_lag_features(df, col, lags):
#     for lag in lags:
#         df[f'{col}_lag_{lag}'] = df[col].shift(lag)
#     return df

# lags = [1, 2, 3, 4, 5]
# df = create_lag_features(df, 'rate', lags)

# df = df.dropna().reset_index(drop=True)

# ----------------------------
# 3. Definir X e y
# ----------------------------

# Para prever: mean_1 = média próxima janela
# Aqui, vamos fingir que o 'rate' futuro é deslocado
df['mean_1'] = df['rate'].shift(-1)
df['stdev_1'] = df['rate'].rolling(window=2).std().shift(-1)
df['mean_2'] = df['rate'].shift(-2)
df['stdev_2'] = df['rate'].rolling(window=3).std().shift(-2)

# Remover NaNs criados pelo shift para targets
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=['id', 'rate', 'mean_1', 'stdev_1', 'mean_2', 'stdev_2'])
y_mean_1 = df['mean_1']
y_stdev_1 = df['stdev_1']
y_mean_2 = df['mean_2']
y_stdev_2 = df['stdev_2']

# ----------------------------
# 4. Criar pipeline base
# ----------------------------

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, None]
}

tscv = TimeSeriesSplit(n_splits=5)

# ----------------------------
# 5. Treinar modelo para cada target
# ----------------------------

def train_and_predict(X, y, name):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1
    )
    grid.fit(X, y)
    print(f"{name} | Melhor MAPE: {grid.best_score_:.4f} | Melhor params: {grid.best_params_}")
    return grid.predict(X)

# Prever cada target
mean_1_pred = train_and_predict(X, y_mean_1, 'mean_1')
stdev_1_pred = train_and_predict(X, y_stdev_1, 'stdev_1')
mean_2_pred = train_and_predict(X, y_mean_2, 'mean_2')
stdev_2_pred = train_and_predict(X, y_stdev_2, 'stdev_2')

# ----------------------------
# 6. Gerar submissão final
# ----------------------------

submission = pd.DataFrame({
    'id': df['id'],
    'mean_1': mean_1_pred,
    'stdev_1': stdev_1_pred,
    'mean_2': mean_2_pred,
    'stdev_2': stdev_2_pred
})

submission.to_csv('submission.csv', index=False)
print("Submissão salva: submission.csv")

