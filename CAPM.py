import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st


# 1. Récupération des données (Action vs Indice de marché)
tickers = ["GLE.PA", "^FCHI"]  # GLE = SocGen, ^FCHI = CAC 40
data = yf.download(tickers, start="2010-01-01", end="2026-02-01")['Close']

# 2. Calcul des rendements logarithmiques (plus stables pour la régression)
returns = data.pct_change().dropna()
returns.columns = ['CAC40', 'SocGen']

# 3. Paramètres du modèle
# On prend un taux sans risque (Risk-free rate) théorique à 3% annuel (0.03)
rf_annual = 0.03
rf_daily = (1 + rf_annual)**(1/252) - 1

# Calcul des primes de risque (Excess Returns)
returns['Excess_SocGen'] = returns['SocGen'] - rf_daily
returns['Excess_Market'] = returns['CAC40'] - rf_daily

# 4. Régression Linéaire pour trouver le Bêta
# Y = alpha + beta * X
X = sm.add_constant(returns['Excess_Market']) # Ajoute l'ordonnée à l'origine (Alpha)
Y = returns['Excess_SocGen']

model = sm.OLS(Y, X).fit()
alpha, beta = model.params

print(f"Bêta de la Société Générale : {beta:.4f}")
print(f"Alpha (Surperformance) : {alpha:.6f}")

# 5. Visualisation de la Droite de Marché (SML)
plt.figure(figsize=(10, 6))
plt.scatter(returns['Excess_Market'], returns['Excess_SocGen'], alpha=0.3, label="Données quotidiennes")
plt.plot(returns['Excess_Market'], alpha + beta * returns['Excess_Market'], color='red', label=f"Droite de régression (β={beta:.2f})")
plt.title("Modèle CAPM : Société Générale vs CAC 40")
plt.xlabel("Prime de risque du Marché (CAC 40)")
plt.ylabel("Prime de risque de l'Action (GLE)")
plt.legend()
plt.grid(True)
plt.show()

#Test de fisher 
# Après avoir ajusté ton modèle : model = sm.OLS(Y, X).fit()
# Extraction des statistiques de Fisher
f_stat = model.fvalue
f_pvalue = model.f_pvalue

print(f"Statistique de Fisher (F-stat) : {f_stat:.2f}")
print(f"P-value du test de Fisher      : {f_pvalue:.4e}")

if f_pvalue < 0.05:
    print("Résultat : Le modèle est globalement significatif (on rejette H0).")
else:
    print("Résultat : Le modèle n'est pas significatif.")

# Analyse des résidus
plt.figure(figsize=(8, 5))
plt.hist(model.resid, bins=50, edgecolor='black')
plt.title("Distribution des résidus du modèle CAPM")
plt.show()


