from scipy.optimize import minimize

# Fonction objectif
f = lambda x: x[0]**2 - 2*x[1]

# Point de départ
x0 = (1.0, 0.0)

# Optimisation
res = minimize(
    f,
    x0,
    tol=1e-6,
    constraints=(
        {'type': 'eq', 'fun': lambda x: x[0]**2 - 2*x[1]} # L'expression doit être égale à 0
    ),
    method='SLSQP' 
)

print(f"Succès : {res.success}")
print(f"Coordonnées optimales (x, y) : {res.x}")
print(f"Valeur minimale de f : {res.fun}")

#Fonction pour minimiser des écarts
#On minimise la somme des erreurs au carré
import numpy as np
from scipy.optimize import minimize

# Génération de données (y = 10 + 15*X1 + 20*X2)
np.random.seed(42)
x_1 = np.linspace(0, 10, 25)
x_2 = np.linspace(0, 10, 25)
# Ajout d'un petit bruit aléatoire 
y_data = 10 + 15 * x_1 + 20 * x_2 + np.random.normal(0, 1, 20)

# Définition de la fonction SSE
def sse_function(params, x1, x2, y):
    # params contient [constante, coeff_x1, coeff_x2]
    b0, b1, b2 = params
    predictions = b0 + b1 * x1 + b2 * x2
    erreurs = y - predictions
    return np.sum(erreurs**2)

# Point de départ
initial_guess = [1.0, 1.0, 1.0]

# Optimisation
res = minimize(sse_function, initial_guess, args=(x_1, x_2, y_data))


b0_opt, b1_opt, b2_opt = res.x

print(f"Succès : {res.success}")
print(f"Intercept (b0) : {b0_opt:.4f}")
print(f"Coeff X1 (b1)  : {b1_opt:.4f}")
print(f"Coeff X2 (b2)  : {b2_opt:.4f}")
#Nelson Siege

#utilisation de Parquet avec les données funds
