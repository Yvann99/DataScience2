#Bibliothèques
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Data
tickers = {"GLE.PA","^FCHI"}
data_set = yf.download(tickers, period = "3Y")["Close"]
data_set_clean = data_set.dropna()
#calcul des log returns
log_returns = np.log(data_set_clean/data_set_clean.shift(1)).dropna()
log_returns.columns = [f"Log_Ret_{col}" for col in log_returns.columns]

resultat = pd.concat([data_set_clean, log_returns], axis=1).dropna()
print(resultat.head())
#Standardisation
#Calcul du z score
returns_std = (log_returns - log_returns.mean())/log_returns.std()
returns_std.columns = [f"Std_Ret_{col}" for col in log_returns.columns]
print(returns_std.head())

#Normalisation min/max
returns_min = log_returns.min()
returns_max = log_returns.max()

returns_norm = (log_returns - returns_min)/(returns_max - returns_min)
returns_norm.columns = [f"Norm_Ret_{col}" for col in log_returns.columns]
print(returns_norm.head())

#On passe au train/test split
from sklearn.model_selection import train_test_split

# On définit les variables (X = CAC 40, y = GLE.PA par exemple)
X = log_returns[['Log_Ret_^FCHI']] # Variables explicatives (Indice)
y = log_returns['Log_Ret_GLE.PA']   # Variable cible (Action)

# Split : 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Taille du Train : {len(X_train)} points")
print(f"Taille du Test : {len(X_test)} points")

from sklearn.linear_model import LinearRegression

# 1. Création du modèle
model = LinearRegression()

# 2. Entraînement sur les données de Train
# Rappel : X_train contient le CAC 40 et y_train contient GLE.PA
model.fit(X_train, y_train)

# 3. Extraction des coefficients
beta = model.coef_[0]
alpha = model.intercept_
r_carre = model.score(X_train, y_train)

print("-" * 30)
print(f"RÉSULTATS DE LA RÉGRESSION (TRAIN)")
print("-" * 30)
print(f"Bêta (sensibilité) : {beta:.4f}")
print(f"Alpha (ordonnée à l'origine) : {alpha:.4f}")
print(f"R² (qualité de l'ajustement) : {r_carre:.4f}")

from sklearn.metrics import mean_squared_error, r2_score

# 1. Prédiction sur les données de test
y_pred = model.predict(X_test)

# 2. Calcul des métriques
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)

print("-" * 30)
print(f"ÉVALUATION SUR LE TEST SET (20%)")
print("-" * 30)
print(f"R² (Test) : {r2_test:.4f}")
print(f"RMSE : {rmse_test:.4f}")

# 3. Comparaison Train vs Test (détection d'overfitting)
print(f"Différence R² (Train - Test) : {r_carre - r2_test:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Réel (GLE.PA)", color='blue', alpha=0.7)
plt.plot(y_pred, label="Prédit (via CAC 40)", color='orange', linestyle='--')
plt.title("Comparaison Rendements Réels vs Prédits (Test Set)")
plt.legend()
plt.show()