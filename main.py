#Bibliothèques
import yfinance as yf
import numpy as np
import pandas as pd
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
X = log_returns[['^FCHI']]
y = log_returns['GLE.PA']

# Split : 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Taille du Train : {len(X_train)} points")
print(f"Taille du Test : {len(X_test)} points")