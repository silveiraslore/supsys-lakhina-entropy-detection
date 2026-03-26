# supsys-lakhina-entropy-detection

Projet académique IMT Atlantique autour de la détection de botnets sur CTU-13
avec une approche inspirée de Lakhina Entropy.

## Structure

- `main_exploration.py` : exploration, visualisations et création des splits
- `main_detection.py` : entraînement, calibration, évaluation et sauvegarde
- `preprocessing/` : chargement, nettoyage et gestion des splits
- `analysis/` : statistiques et graphiques d'exploration
- `detection/` : détecteur Lakhina Entropy
- `evaluation/` : métriques et figures d'évaluation
- `main.ipynb` : notebook interactif reprenant le pipeline

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Télécharger le scénario 9 de CTU-13 puis placer le fichier ici :

```text
dataset/9/capture20110817.binetflow
```

## Exécution

Exploration et génération des splits :

```bash
python3 main_exploration.py
```

Détection complète :

```bash
python3 main_detection.py
```

Pipeline complet via le point d'entrée unique :

```bash
python3 main.py
```

## Notes

- Les splits `train / val / test` sont temporels.
- Une métadonnée est sauvegardée avec les splits pour invalider les versions
  obsolètes si le prétraitement change.
- L'évaluation est faite sur des observations agrégées
  `fenêtre temporelle × IP source`, pas sur le flow individuel.
