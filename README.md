# Predictive Maintenance — Industrial Mining Equipment

Projeto de ciência de dados para manutenção preditiva de equipamentos industriais na área de mineração, utilizando dados de sensores (temperatura, vibração, torque, desgaste) para prever falhas antes que ocorram.

## Dataset
- **UCI AI4I 2020 Predictive Maintenance Dataset**
- 10.000 registros, 6 features originais, 5 tipos de falha
- [Link para download](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Pipeline
1. Análise exploratória (EDA)
2. Feature engineering (potência, diferença de temperatura, desgaste acumulado)
3. Balanceamento com SMOTE
4. Modelo Random Forest com validação cruzada
5. Avaliação: ROC-AUC, matriz de confusão, feature importance

## Estrutura
```
predictive-maintenance/
├── data/               # Dados brutos e processados
├── src/                # Scripts Python
├── outputs/plots/      # Visualizações geradas
├── requirements.txt
└── README.md
```

## Como rodar
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/features.py
python src/model.py
```

## Tecnologias
Python · Pandas · Scikit-learn · Imbalanced-learn · Seaborn · Matplotlib · Joblib
