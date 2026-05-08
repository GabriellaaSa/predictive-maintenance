import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Carregamento ──────────────────────────────────────────────
df = pd.read_csv("data/ai4i2020.csv")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── Visão geral ───────────────────────────────────────────────
print("\nValores nulos:\n", df.isnull().sum())
print("\nDistribuição da variável alvo (Machine failure):")
print(df["Machine failure"].value_counts(normalize=True).round(3))

# ── Renomeia colunas (remove espaços) ─────────────────────────
df.columns = [c.strip().replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]

# ── Variáveis numéricas ───────────────────────────────────────
numeric_cols = ["Air_temperature_K", "Process_temperature_K",
                "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(data=df, x=col, hue="Machine_failure", kde=True, ax=ax, palette="Set2")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("outputs/plots/distributions.png", dpi=150)
plt.show()

# ── Correlação ────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ["Machine_failure"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("outputs/plots/correlation.png", dpi=150)
plt.show()

# ── Tipos de falha ────────────────────────────────────────────
failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_counts = df[failure_types].sum().sort_values(ascending=False)
plt.figure(figsize=(7, 4))
sns.barplot(x=failure_counts.index, y=failure_counts.values, palette="Reds_r")
plt.title("Frequência por Tipo de Falha")
plt.ylabel("Ocorrências")
plt.tight_layout()
plt.savefig("outputs/plots/failure_types.png", dpi=150)
plt.show()