#Bibliothèques
import yfinance as yf
import numpy as np
import pandas as pd
#Data
tickers = {"GLE.PA","^FCHI"}
data_set = yf.download(tickers, period = "3Y")["Close"]
data_set_clean = data_set.dropna()
#calcul des log returns
log_returns = np.log(data_set_clean/data_set_clean.shift(1))
log_returns.columns = [f"Log_Ret_{col}" for col in log_returns.columns]

resultat = pd.concat([data_set_clean, log_returns], axis=1).dropna()
print(resultat.head())