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