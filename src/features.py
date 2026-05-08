import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Diferença de temperatura (processo - ambiente) ────────
    df["temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]

    # ── Potência estimada ─────────────────────────────────────
    # P = torque × velocidade angular (rad/s)
    df["power_W"] = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)

    # ── Desgaste × torque (proxy de estresse acumulado) ───────
    df["wear_torque"] = df["Tool_wear_min"] * df["Torque_Nm"]

    # ── Temperatura normalizada pelo RPM ──────────────────────
    df["temp_per_rpm"] = df["Process_temperature_K"] / (df["Rotational_speed_rpm"] + 1)

    # ── Flag de desgaste crítico (> 200 min) ──────────────────
    df["high_wear"] = (df["Tool_wear_min"] > 200).astype(int)

    # ── Encoder do tipo de produto ────────────────────────────
    df["product_type_enc"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/ai4i2020.csv")
    df.columns = [c.strip().replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]
    df = build_features(df)
    print(df[["temp_diff", "power_W", "wear_torque", "high_wear"]].describe())
    df.to_csv("data/ai4i2020_features.csv", index=False)
    print("✅ Features salvas em data/ai4i2020_features.csv")