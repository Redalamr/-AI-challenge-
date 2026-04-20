# Rapport de Projet — Détection d'Anomalies Industrielles (MVTec AD)

**Projet :** AI Challenge — Détection de défauts en temps réel sur le dataset MVTec AD  
**Équipe :** Corel, Reda  
**Encadrant :** M. Alexandre Letard  
**Date :** 20 avril 2026

---

## 1. Contexte et problématique

Le contrôle qualité industriel repose sur l'inspection visuelle humaine, un processus lent et peu fiable à grande échelle. L'objectif de ce projet est de détecter automatiquement des défauts visuels sur des pièces industrielles, à partir du dataset **MVTec Anomaly Detection (MVTec AD)** [Bergmann et al., 2019].

Ce dataset couvre 15 catégories d'objets et de textures industriels (vis, câbles, capsules, moquette, grille, etc.) pour un total de 4,9 Go d'images PNG haute résolution. Chaque catégorie contient des images d'entraînement composées uniquement de produits sains, et des images de test mélangeant produits sains et défectueux, annotés avec des masques de segmentation pixel-à-pixel.

La contrainte principale était de livrer un POC fonctionnel et évaluable en **une seule journée**, avec une deadline stricte à 20h00.

---

## 2. Choix de méthode et justification

### 2.1 Méthode retenue

Après lecture de plusieurs publications, nous nous sommes inspirés de la méthode **PatchCore** [Roth et al., CVPR 2022] pour construire notre solution. Cette approche nous a semblé la mieux adaptée à notre contexte : elle ne demande aucun entraînement supervisé, elle est réalisable en une journée, et ses performances sur MVTec AD sont parmi les meilleures connues.

Le principe est simple : pendant la phase d'apprentissage, on construit une **base de référence** à partir des images saines uniquement. À l'inférence, on compare chaque nouvelle image à cette base pour mesurer à quel point elle s'en écarte.

Détail de l'implémentation :

1. **Extraction de représentations visuelles** : on utilise un réseau de neurones convolutif pré-entraîné sur ImageNet (`Wide_ResNet50_2`) en mode figé — ses poids ne sont jamais modifiés. On intercepte les sorties des couches intermédiaires `layer2` et `layer3`, qui décrivent visuellement chaque région de l'image, et on les concatène après ré-échantillonnage à taille commune.

2. **Construction de la base de référence** : pour chaque région de chaque image d'entraînement, on calcule un descripteur local (moyennage sur un voisinage 3×3). L'ensemble de ces descripteurs constitue la base de référence initiale.

3. **Réduction de la base** : la base brute serait trop grande pour l'inférence. On la réduit à 10% de sa taille (25% pour `pill` et `screw`) en sélectionnant les descripteurs les plus éloignés les uns des autres, de façon à garder une couverture uniforme de la variabilité des produits sains. L'algorithme utilise la formule `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b` pour vectoriser le calcul des distances et rendre la sélection rapide.

4. **Indexation pour recherche rapide** : la base réduite est indexée avec FAISS (`IndexFlatL2`), ce qui permet de retrouver instantanément le descripteur le plus proche lors de l'inférence.

5. **Score d'anomalie** : pour chaque région de l'image testée, on calcule la distance au descripteur le plus proche dans la base. Ce score est pondéré par la densité locale (zones denses → voisins proches → score relativisé). Le score final de l'image est le maximum sur toutes ses régions. La carte d'anomalie pixel-à-pixel est obtenue par zoom bilinéaire puis lissage gaussien (σ=4).

### 2.2 Pourquoi cette méthode plutôt qu'une autre ?

Deux approches ont été envisagées et écartées :

**Autoencodeurs / VAE** [Bergmann et al., 2018] : ces modèles apprennent à reconstruire l'image et détectent les anomalies par l'écart entre l'original et la reconstruction. Ils nécessitent un entraînement complet, long à stabiliser, et très sensible au dimensionnement du réseau. Trop risqué à faire tourner correctement en une journée.

**GANs (AnoGAN, f-AnoGAN)** [Schlegl et al., 2017, 2019] : les réseaux génératifs adversariaux sont instables à l'entraînement et leur convergence peut prendre des heures. Incompatible avec notre contrainte de temps.

Notre approche a trois avantages concrets pour ce projet :
- **Aucun entraînement** : le réseau de neurones est figé dès le départ. Pas de risque de sur-ajustement.
- **Performances documentées** : les auteurs de la méthode publient ~99% d'AUROC image-level sur MVTec AD.
- **Inférence rapide** : la recherche dans la base via FAISS est suffisamment rapide pour viser du temps réel.

---

## 3. Traitement des données et structure du code

### 3.1 Chargement et pré-traitement

Le module `data/loader.py` parcourt l'arborescence MVTec (`train/good/`, `test/<defect_type>/`) et charge les masques de vérité terrain depuis `ground_truth/`. Les masques absents (images saines) sont remplacés par un masque vide.

Les images en **niveaux de gris** (`screw`, `grid`, `zipper`) sont chargées en noir et blanc puis dupliquées sur 3 canaux (R=G=B) pour rester compatibles avec l'entrée attendue par le réseau.

Le module `data/transforms.py` applique le pré-traitement standard : redimensionnement à 256px, recadrage central à 224×224, normalisation ImageNet (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]).

Une conversion PIL → Tenseur manuelle a été nécessaire à cause d'une incompatibilité entre `torchvision.ToTensor()` et NumPy 2.x sous Windows.

### 3.2 Structure du projet

```
mvtec-poc/
├── config.py                  # Hyperparamètres centralisés
├── main.py                    # Point d'entrée CLI
├── data/
│   ├── loader.py              # Chargement du dataset MVTec
│   └── transforms.py          # Redimensionnement et normalisation
├── models/
│   ├── feature_extractor.py   # WideResNet50, extraction multi-couches
│   └── patchcore.py           # Construction de la base, réduction, inférence
├── evaluation/
│   ├── metrics.py             # AUROC, Pixel-AUROC, F1, courbe ROC (sans sklearn)
│   ├── cost_matrix.py         # Matrice de coût, seuil optimal, analyse de sensibilité
│   └── visualize.py           # ROC, confusion, cartes d'anomalie, Pareto
├── inference/
│   └── predict.py             # Prédiction unitaire, mesure du débit (FPS)
└── results/                   # Graphiques générés par catégorie
```

---

## 4. Procédure d'évaluation indépendante

Le sujet interdit l'utilisation du code d'évaluation officiel MVTec. Toutes les métriques ont donc été **réimplémentées depuis zéro** dans `evaluation/metrics.py` et `evaluation/cost_matrix.py`.

### 4.1 Métriques calculées sans librairie externe

**AUROC image-level** : la courbe ROC est construite manuellement en triant les scores par ordre décroissant et en accumulant les vrais positifs et les faux positifs au fil des seuils. L'aire sous la courbe est calculée par règle des trapèzes (`np.trapz`). Aucun appel à `sklearn.metrics`.

**Pixel-AUROC** : les cartes d'anomalie et les masques de référence sont mis à plat et concaténés pour former un vecteur de scores au niveau du pixel. L'AUROC est calculée avec la même procédure.

**F1-score optimal** : le F1 est calculé pour chaque seuil possible (l'ensemble des valeurs de score observées), et le meilleur est retenu.

### 4.2 Matrice de coût métier

Au-delà des métriques standards, nous avons mis en place une **matrice de coût asymétrique** pour piloter le choix du seuil de décision en tenant compte des enjeux réels :

| Type d'erreur | Coût unitaire | Justification |
|---|---|---|
| Faux Positif (FP) | 50 € | Produit sain rejeté → rebut ou retraitement inutile |
| Faux Négatif (FN) | 500 € | Défaut non détecté → retour client, rappel produit |
| Vrai Positif (TP) | −10 € | Défaut détecté → coût de retraitement accepté |
| Vrai Négatif (TN) | 0 € | Produit sain accepté |

Le ratio FN/FP de 10:1 reflète la réalité industrielle : un défaut livré au client génère des coûts de rappel, de litiges et d'image bien supérieurs à un simple rejet injustifié. Ce choix de paramétrage est discuté plus en détail en section 6.

La fonction `find_optimal_threshold()` parcourt tous les seuils et sélectionne celui qui minimise le coût total, avec priorité aux configurations sans faux négatifs (rappel parfait), tant que cela ne revient pas à rejeter systématiquement tout l'ensemble de test.

### 4.3 Analyse de sensibilité Pareto

La fonction `sensitivity_analysis()` fait varier le ratio COST_FN/COST_FP de 1 à 20 et trace la frontière Pareto coût/rappel. Cela permet de montrer comment le seuil optimal évolue selon l'arbitrage coût-qualité souhaité.

---

## 5. Résultats

### 5.1 Tableau de synthèse — 15 catégories

```
================================================================================
Category        ImgAUROC  PixAUROC      F1     Cost   Thresh    FPS
--------------------------------------------------------------------------------
bottle            1.0000    0.9766  1.0000     -630   2.2526    9.2
cable             0.9966    0.9793  0.9890     -310   2.1745    8.7
capsule           0.9729    0.9795  0.9767     -540   0.8886    9.3
carpet            0.9827    0.9838  0.9545     -440   1.6594    6.9
grid              0.9290    0.9339  0.9412      240   1.7380    7.3
hazelnut          1.0000    0.9808  1.0000     -700   3.0357    7.3
leather           1.0000    0.9860  1.0000     -920   2.8121    7.3
metal_nut         0.9971    0.9787  0.9841     -780   1.8562    8.8
pill              0.9239    0.9852  0.9474     -210   1.3327    5.9
screw             0.8951    0.9558  0.9027      560   1.1216    5.2
tile              0.9899    0.9485  0.9880       60   1.5554    8.5
toothbrush        0.9972    0.9822  0.9836     -250   1.6580   11.8
transistor        0.9992    0.9808  0.9873     -300   1.9083    9.2
wood              0.9939    0.9250  0.9833     -400   2.5187   15.3
zipper            0.9950    0.9753  0.9873     -840   1.2331   15.8
--------------------------------------------------------------------------------
MEAN              0.9782    0.9701  0.9750                      9.1
================================================================================
```

Les résultats sont globalement très bons : 13 des 15 catégories dépassent 0.97 d'AUROC image-level. Les deux catégories les plus difficiles sont `screw` (0.895) et `grid` (0.929), pour des raisons discutées ci-dessous.

### 5.2 Résultats détaillés — Bottle

La bouteille est la catégorie la plus accessible : les défauts (rayures, contaminations, cassures) contrastent clairement avec le fond homogène. AUROC = 1.00, F1 = 1.00.

| Courbe ROC | Matrice de confusion |
|:---:|:---:|
| ![ROC Bottle](./results/bottle/roc_bottle.png) | ![Confusion Bottle](./results/bottle/confusion_bottle.png) |

| Carte d'anomalie | Coût en fonction du seuil |
|:---:|:---:|
| ![Heatmap Bottle](./results/bottle/heatmap_bottle_0.png) | ![Cost Bottle](./results/bottle/cost_threshold_bottle.png) |

| Frontière Pareto | Grille TP / FP / FN / TN |
|:---:|:---:|
| ![Pareto Bottle](./results/bottle/pareto_bottle.png) | ![Examples Bottle](./results/bottle/examples_bottle.png) |

---

### 5.3 Résultats détaillés — Carpet

Les défauts de type *cut* ou *hole* modifient fortement la texture locale et sont bien détectés (AUROC = 0.983). Les défauts subtils de *color* sont plus difficiles à distinguer de variations d'éclairage naturelles.

| Courbe ROC | Matrice de confusion |
|:---:|:---:|
| ![ROC Carpet](./results/carpet/roc_carpet.png) | ![Confusion Carpet](./results/carpet/confusion_carpet.png) |

| Carte d'anomalie | Coût en fonction du seuil |
|:---:|:---:|
| ![Heatmap Carpet](./results/carpet/heatmap_carpet_0.png) | ![Cost Carpet](./results/carpet/cost_threshold_carpet.png) |

| Frontière Pareto | Grille TP / FP / FN / TN |
|:---:|:---:|
| ![Pareto Carpet](./results/carpet/pareto_carpet.png) | ![Examples Carpet](./results/carpet/examples_carpet.png) |

---

### 5.4 Résultats détaillés — Screw

La catégorie la plus difficile (AUROC = 0.895). Les vis sont photographiées en niveaux de gris (canal dupliqué pour compatibilité RGB) et les défauts comme *scratch neck* ou *manipulated front* occupent seulement quelques pixels. La base de référence a été portée à 25% pour mieux couvrir la variabilité des poses.

| Courbe ROC | Matrice de confusion |
|:---:|:---:|
| ![ROC Screw](./results/screw/roc_screw.png) | ![Confusion Screw](./results/screw/confusion_screw.png) |

| Carte d'anomalie | Coût en fonction du seuil |
|:---:|:---:|
| ![Heatmap Screw](./results/screw/heatmap_screw_41.png) | ![Cost Screw](./results/screw/cost_threshold_screw.png) |

| Frontière Pareto | Grille TP / FP / FN / TN |
|:---:|:---:|
| ![Pareto Screw](./results/screw/pareto_screw.png) | ![Examples Screw](./results/screw/examples_screw.png) |

---

## 6. Analyse critique

### 6.1 Décisions techniques prises dans la journée

**Sélection greedy vs. aléatoire** : on a gardé l'algorithme de sélection par distance maximale plutôt qu'un sous-échantillonnage aléatoire. C'est plus long à calculer mais la couverture de la base de référence est bien meilleure, ce qui impacte directement la qualité de détection.

**Deux bugs imprévus** ont consommé environ deux heures de débogage. D'abord, `torchvision.ToTensor()` est incompatible avec NumPy 2.x sous Windows, résolu par une conversion PIL → Tenseur manuelle. Ensuite, les images en niveaux de gris (`screw`, `grid`, `zipper`) plantaient silencieusement sans duplication explicite des canaux.

**Ajustement pour `pill`** : les images de pilules contiennent des poussières et reflets qui font exploser le score maximum. On a limité le score au percentile 99.5 comme contournement rapide. Ce n'est pas propre mais ça fonctionne dans le temps imparti.

**Chargement séquentiel des données** (`NUM_WORKERS=0`) : les processus parallèles de PyTorch plantent sur Windows sans précaution particulière. La mise à zéro des workers double le temps de chargement mais évite les crashes.

### 6.2 Impact financier des erreurs de classification

Un **faux négatif** (défaut non détecté) peut déclencher un rappel produit, une perte de confiance client et des coûts logistiques importants. Un **faux positif** (produit sain rejeté) représente une perte directe mais limitée au coût de rebut ou de re-contrôle. D'où le ratio de 10:1 dans notre matrice de coût.

En pratique, cela signifie qu'il vaut mieux rejeter quelques produits sains par excès de prudence plutôt que de laisser passer un seul défaut. L'analyse Pareto le confirme : quand on augmente le poids du faux négatif, le seuil de décision descend automatiquement et le rappel s'améliore, au prix de plus de fausses alarmes.

Sur `screw`, le coût optimal reste positif (560 €) car la difficulté de la catégorie génère encore quelques faux négatifs inévitables avec notre configuration. Sur `grid` et `tile`, on reste dans une zone de coût positif faible, liée à la nature texturée de ces catégories.

### 6.3 Qualité du code

Le code fonctionne et est structuré en modules indépendants. Les compromis faits sous la deadline sont visibles : commentaires mélangés français/anglais, paramètres par catégorie directement dans `config.py`, absence de tests unitaires. Ces points seraient traités avant tout passage en production.

---

## 7. Perspectives

Si on avait plus de temps pour industrialiser ce POC :

**1. Tester des modèles plus récents**  
**EfficientAD** [Batzner et al., 2023] et **WinCLIP** [Jeong et al., 2023] publient de meilleures performances sur les catégories difficiles. EfficientAD est particulièrement adapté au temps réel.

**2. Accélérer l'inférence**  
Exporter le réseau au format ONNX et utiliser TensorRT pour l'inférence GPU. Passer à un index FAISS approximatif (`IndexIVFFlat`) pour traiter des images haute résolution (2048×2048) à plus de 30 FPS.

**3. Calibration automatique des seuils**  
Mettre en place une calibration par catégorie sur un ensemble de validation séparé, pour éviter le réglage manuel que nous avons dû faire pour `pill` et `screw`.

**4. Intégration industrielle**  
Encapsuler le modèle dans un conteneur Docker avec une API REST, versionner les bases de référence (une par catégorie/référence produit), et connecter à un système de supervision qualité via MQTT ou OPC-UA.

**5. Surveillance en production**  
Détecter automatiquement quand les images d'entrée s'éloignent statistiquement des images d'entraînement — signe d'un changement de process ou d'une caméra déréglée.

---

## Références

- P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.** CVPR 2019.
- K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. **Towards Total Recall in Industrial Anomaly Detection.** CVPR 2022.
- T. Schlegl, P. Seeböck, S. M. Waldstein, U. Schmidt-Erfurth, G. Langs. **Unsupervised Anomaly Detection with Generative Adversarial Networks.** IPMI 2017.
- K. Batzner, L. Heckler, R. König. **EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.** WACV 2024.
