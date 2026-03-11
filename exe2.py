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

#Newton Raphson
def f(x):
    return x**2 + 10*x - 65 

def f1(x):
    return 2*x + 10

def NR(x0, f, n):
    x = x0  # On commence à la valeur initiale x0
    C = [x] # On stocke la valeur de départ
    for i in range(n):
        # Formule : x_n+1 = x_n - f(x_n) / f'(x_n)
        x = x - f(x) / f1(x)
        C.append(x)
    return C
resultats = NR(5, f, 5) # Point de départ 5, sur 5 itérations
print(resultats)

from s1_dichotomie import dichotomie
from s2_newton_raphson import newton_raphson

def f(r):
    return 30000.0/(1+r) \
           +40000.0/(1+r)**2 \
           +50000.0/(1+r)**3 \
           -100000.0

r, fr, i=dichotomie(f, 0, 1, 0, tol=1e-6)
print(f"Solution approchée: r={r}, f(r)={fr}, itérations={i}")

def df_approche_finie(f, x):
    h = 1e-5
    return (f(x+h) - f(x)) / h

def df(r):
    return df_approche_finie(f, r)

r0 = 0.1
r = newton_raphson(f, df, r0)
print(f"Solution approchée: r={r}, f(r)={f(r)}")