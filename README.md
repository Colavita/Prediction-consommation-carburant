# Prédiction de la consommation de carburant

## Introduction

Ce projet a été réalisé dans le cadre du cours **MTH3302 - Méthodes probabilistes et statistiques pour l'I.A.** à Polytechnique Montréal (session A2024). L'objectif est de prédire la consommation d'essence (L/100km) de véhicules récents à partir de leurs caractéristiques techniques, en utilisant différentes méthodes statistiques et d'apprentissage automatique.

## Objectif

Prédire la consommation d'essence pour un ensemble de véhicules de validation, à partir de leurs caractéristiques, en testant différentes approches de traitement de données et de modélisation.

## Données

Les données sont situées dans le dossier `data/raw/` :
- **train.csv** : Données d'entraînement avec les caractéristiques et la consommation d'essence.
- **test.csv** : Données de test (validation) avec les mêmes caractéristiques, sans la consommation.

**Colonnes principales :**
- `annee` : Année de fabrication
- `type` : Type de véhicule (ex: VUS_petit, voiture_moyenne, etc.)
- `nombre_cylindres` : Nombre de cylindres du moteur
- `cylindree` : Cylindrée du moteur (L)
- `transmission` : Type de transmission (traction, intégrale, propulsion, 4x4)
- `boite` : Type de boîte de vitesses (automatique, manuelle)
- `consommation` : Consommation d'essence (L/100km, seulement dans train.csv)

## Méthodologie

Le projet explore plusieurs approches :
- **Exploration et prétraitement des données** :
  - Encodage one-hot des variables catégorielles
  - Traitement des valeurs manquantes et aberrantes (outliers)
  - Transformation logarithmique de certaines variables numériques
  - Regroupement de catégories pour réduire la dimensionnalité
- **Modélisation** :
  - Régression linéaire (modèle principal retenu)
  - Régression bayésienne (Ridge)
  - Régression par composantes principales (PCA)
- **Validation** :
  - Séparation train/validation (80/20)
  - Validation croisée (k-fold)
  - Évaluation par RMSE, BIC, R² ajusté

Le meilleur modèle final est une régression linéaire utilisant :
- Transmission (traction, intégrale)
- Catégories de véhicules (petits, moyens, grands)
- Cylindrée

## Structure du projet

```
.
├── data/
│   ├── raw/         # Données brutes (train.csv, test.csv)
│   └── processed/   # Données prétraitées (optionnel)
├── notebooks/
│   └── projet_equipe_R.ipynb  # Notebook principal (Julia)
├── submissions/
│   ├── linear/      # Soumissions modèles linéaires
│   ├── bayes/       # Soumissions modèles bayésiens
│   └── pca/         # Soumissions PCA
└── .gitignore
```

## Comment exécuter le projet

1. **Prérequis** :
   - Julia (>= 1.10 recommandé)
   - Packages Julia : CSV, DataFrames, Statistics, Gadfly, Plots, GLM, MultivariateStats, etc.
2. Ouvrir le notebook `notebooks/projet_equipe_R.ipynb` dans Jupyter ou Pluto.jl.
3. Exécuter les cellules pour reproduire l'analyse, l'entraînement et la génération des prédictions.
4. Les résultats (prédictions) sont exportés dans le dossier `submissions/`.

## Résultats

- **Meilleur RMSE obtenu (validation interne)** : ~0.83

## Auteurs / Crédits

Projet réalisé par une équipe d'étudiants de Polytechnique Montréal dans le cadre du cours MTH3302.

## Remarques

- Pour améliorer les résultats, il serait possible d'explorer des modèles plus avancés (XGBoost, forêts aléatoires), d'enrichir le prétraitement ou d'augmenter la taille des données.
- Voir le notebook pour tous les détails, analyses et justifications des choix méthodologiques. 