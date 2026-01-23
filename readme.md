##création d'un environnement virtuel avec venv

pyton-m venv .venv

#installation d'un package

pip install pandas

##freeze des packages (écriture du fichier)

pip freeze > requirements.txt

##synchronisation des librairies

pip install -r requirements.txt

#installation d'un environnement avec uv

dans le terminal avec curl -LsSf https://astral.sh/uv/install.sh | sh

#crée un nouveau projet (fermer le terminal dans VS code pour actualiser le path)

uv init

##synchronisation de l'environnement

uv sync
## installation de streamlit

uv add streamlit