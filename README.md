# supsys-lakhina-entropy-detection
Academic project for the course "Supervision des Systèmes et Audit de Sécurité" at IMT Atlantique, focused on implementing and evaluating a Lakhina entropy-based intrusion detection approach for botnet detection using the CTU-13 dataset.


```

---

## Récapitulatif complet des fichiers du projet

Voici l'arborescence finale avec **tous** les fichiers :
```
project/
│
├── main_exploration.py       ← Étape 1 : exploration du dataset
├── main_detection.py         ← Étape 2 : pipeline complet de détection
│
├── preprocessing/
│   ├── __init__.py
│   └── loader.py
│
├── analysis/
│   ├── __init__.py
│   └── statistics.py
│
├── detection/
│   ├── __init__.py
│   └── lakhina_entropy.py
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
│
├── dataset/
│   └── scenario9/
│       ├── capture20110817.binetflow   ← à télécharger
│       └── splits/                    ← généré automatiquement
│           ├── train.parquet
│           ├── val.parquet
│           └── test.parquet
│
└── results/                           ← généré automatiquement
    ├── predictions.parquet
    ├── metrics.json
    ├── config.json
    ├── metrics_summary.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── score_distribution.png
    ├── metrics_over_time.png
    └── threshold_calibration.png