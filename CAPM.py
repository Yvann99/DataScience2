#On commence par importer les données de SG

import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

ticker = "GLE.PA"
start_date = "2010-10-01"
end_date = "2026-02-01"
df = yf.download(ticker, start = start_date, end = end_date)

print(df.head())
df['Close'].plot(figsize=(10, 6), title=f"Historique du cours {ticker}")
plt.ylabel("Prix en Euros")
plt.grid(True)
plt.show()