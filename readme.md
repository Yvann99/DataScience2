Description du projet
Ce projet analyse la relation entre les rendements de l'action Societe Generale (GLE.PA) et l'indice CAC 40 (^FCHI) sur une periode de 3 ans. L'objectif est de mesurer le risque systematique de l'action par rapport a son marche de reference via une regression lineaire.

Etapes du traitement
Recuperation des donnees historiques via l'API Yahoo Finance.

Calcul des log-rendements pour stabiliser les series temporelles.

Transformation des donnees par standardisation (Z-score) et normalisation (Min-Max).

Division des donnees en ensembles d'entrainement (80%) et de test (20%) sans melange chronologique.

Entrainement d'un modele de regression lineaire pour determiner le Beta et l'Alpha.

Variables et indicateurs
Beta : Mesure la sensibilite de l'action aux mouvements de l'indice.

Alpha : Indique la performance propre de l'action independamment du marche.

R2 : Definit la qualite de l'ajustement du modele aux donnees reelles.

Installation
Le projet necessite Python et les bibliotheques suivantes :

yfinance

pandas

numpy

scikit-learn




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