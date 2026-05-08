import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

# ── Carrega dados com features ────────────────────────────────
df = pd.read_csv("data/ai4i2020_features.csv")

FEATURES = [
    "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
    "temp_diff", "power_W", "wear_torque", "temp_per_rpm",
    "high_wear", "product_type_enc"
]
TARGET = "Machine_failure"

X = df[FEATURES]
y = df[TARGET]

# ── Split estratificado ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Balanceamento com SMOTE (dataset desbalanceado ~3% falhas) ─
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Após SMOTE — Classe 0: {sum(y_train_res==0)}, Classe 1: {sum(y_train_res==1)}")

# ── Pipeline: scaler + modelo ─────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# ── Cross-validation ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train_res, y_train_res,
                             cv=cv, scoring="roc_auc")
print(f"\nROC-AUC CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Treinamento final ─────────────────────────────────────────
pipeline.fit(X_train_res, y_train_res)

# ── Avaliação no test set ─────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Normal", "Falha"]))
print(f"ROC-AUC Test: {roc_auc_score(y_test, y_prob):.4f}")

# ── Matriz de Confusão ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Normal", "Falha"],
    cmap="Blues", ax=axes[0]
)
axes[0].set_title("Matriz de Confusão")

# ── Curva ROC ─────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Curva ROC")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/plots/model_evaluation.png", dpi=150)
plt.show()

# ── Feature Importance ────────────────────────────────────────
importances = pipeline.named_steps["clf"].feature_importances_
feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
import seaborn as sns
sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png", dpi=150)
plt.show()

# ── Salva modelo ──────────────────────────────────────────────
joblib.dump(pipeline, "outputs/model_rf.pkl")
print("\n✅ Modelo salvo em outputs/model_rf.pkl")