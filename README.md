# Portfolio_Analysis

Un Projet Streamlit permettant d'analyser les performances d'un portefeuille financier à l'aide de yfinance, pandas,  plotly, et streamlit. 

##  Quick Start 

`pip install -r requirements.txt`

### Installation

```
git clone https://github.com/ShadyDream222/Portfolio_Analysis.git
cd Portfolio_Analysis
pip install -r requirements.txt
```

### Lancement 

`streamlit run main.py`

## Il possède les fonctionnalités suivantes :

- Sélection d'actifs financiers et récupération des données via Yahoo Finance
- Analyse statistique et graphique des performances du portefeuille
- Calcul d’indicateurs financiers tels que le Sharpe Ratio et la Value at Ris
- Génération automatique de rapports HTML sur la base des données analysées
- Intégration des scores ESG pour une évaluation de l’impact environnemental et social


## Structure du projet : 

├── main.py               # Interface Streamlit
├── functions.py          # Fonctions utilitaires et calculs
├── test_functions.py     # Tests unitaires avec pytest
├── report_template.html  # Template Jinja2 pour les rapports
├── README.md             # Documentation


## Tests unitaires

Les tests unitaires sont définis dans test_functions.py. Pour les exécuter :
`pytest test_functions.py -v --tb=long`

## Quelques images...
