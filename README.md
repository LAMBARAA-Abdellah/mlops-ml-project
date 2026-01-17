# mlops-ml-project 🚀
Mini-projet **Machine Learning baseline** avec **Git & MLOps**  
Dataset : **Iris (scikit-learn)**

---

## 📌 Objectif du projet
Ce projet a pour objectif de mettre en place un **pipeline ML reproductible** intégrant :
- une configuration centralisée (YAML),
- un entraînement et une évaluation de modèle,
- la génération d’artefacts ML,
- un versioning propre avec Git,
- de bonnes pratiques MLOps.

Il a été réalisé dans le cadre de l’atelier **DevOps & MLOps – Mini-projet ML & Git**.

---

## 🗂️ Structure du projet

```text
mlops-ml-project/
├── config/
│   └── train.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── model.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_config.py
├── artifacts/          # (non versionné)
├── notebooks/
├── README.md
└── requirements.txt
```

📌 **Remarque** :  
Les dossiers `artifacts/` et `data/` ne sont pas versionnés conformément aux bonnes pratiques MLOps.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🧪 Entraînement du modèle

```bash
python -m scripts.train
```

### Artefacts générés :
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

---

## 📊 Évaluation du modèle

```bash
python -m scripts.evaluate
```

### Artefact généré :
- `artifacts/report.json`

---

## 📈 Métriques utilisées
- Accuracy
- F1-score (macro)
- Matrice de confusion

---

## 🔁 Reproductibilité
Le projet est entièrement reproductible grâce à :
- la configuration YAML (`config/train.yaml`),
- un pipeline ML déterministe (`random_state` fixé),
- une structure claire et versionnée.

---

## 🧠 Bonnes pratiques MLOps appliquées
- Séparation code / configuration / artefacts
- Versioning Git (code, config, documentation)
- Scripts exécutables (`train`, `evaluate`)
- Artefacts ML générés automatiquement

---

## 👤 Auteur
**Abdellah Lambaraa**  
Étudiant – Master Big Data & Cloud Computing  
Développeur Full Stack

---

## 📜 Licence
Projet académique – usage pédagogique.
