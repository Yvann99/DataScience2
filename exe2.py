#Dichotomie
def dichotomie(f, a, b, target, tol=1e-6, max_iter=100):
    """
    Recherche du zéro de f sur [a, b] par dichotomie
    Args:
        f: fonction dont on cherche le zéro
        a, b: bornes de l'intervalle
        target: valeur cible pour f(c)
        tol: tolérance
        max_iter: nombre maximum d'itérations
    Returns:
        c: solution approchée
    """
    g=lambda x: f(x)-target
    if g(a) * g(b) > 0:
        raise ValueError("g(a) et g(b) doivent être de signes opposés")
    for i in range(max_iter):
        c = (a + b) / 2
        gc = g(c)
        if abs(gc) < tol:
            print(f"Convergence en {i+1} itérations")
            return c, gc, i
        if g(a) * gc < 0:
            b = c
        else:
            a = c
    raise ValueError(f"Pas de convergence après {max_iter} itérations")

import numpy as np

# --- 1. Tes outils d'optimisation ---

def newton_raphson(f, df, x0, n=10): # J'ai réorganisé pour plus de flexibilité
    x = x0
    for i in range(n):
        derivative = df(x)
        if derivative == 0: return x
        x = x - f(x) / derivative
    return x

def dichotomie(f, a, b, tol=1e-6):
    i = 0
    while (b - a) / 2 > tol:
        i += 1
        m = (a + b) / 2
        if f(m) == 0: return m, f(m), i
        if f(a) * f(m) < 0: b = m
        else: a = m
    return (a + b) / 2, f((a + b) / 2), i

# --- 2. Cas concret : Calcul du TRI (Finistech) ---

def f_tri(r):
    # Modèle de flux de trésorerie (Cash Flows)
    return 30000.0/(1+r) + 40000.0/(1+r)**2 + 50000.0/(1+r)**3 - 100000.0

def df_tri(r):
    h = 1e-5
    return (f_tri(r + h) - f_tri(r)) / h

# --- 3. Exécution et Comparaison ---

# Méthode 1 : Dichotomie
r_dicho, fr_dicho, i_dicho = dichotomie(f_tri, 0, 1)
print(f"Dichotomie : r = {r_dicho:.6f} ({i_dicho} itérations)")

# Méthode 2 : Newton-Raphson
r0 = 0.1
r_nr = newton_raphson(f_tri, df_tri, r0, n=5)
print(f"Newton-Raphson : r = {r_nr:.6f} (5 itérations)")