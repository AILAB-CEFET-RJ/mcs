# Adaptive Dengue Forecasting Pipeline

**AutoML Adaptive Hyperparameter Optimization + Dataset Construction for Dengue Case Forecasting**

---

## 📌 Descrição Geral

Pipeline completo e modular para:

- Construção de datasets para previsão de casos de dengue
- Treinamento de modelos (tradicional ou adaptativo)
- Otimização adaptativa de hiperparâmetros
- Registro completo de métricas de desempenho

---

## 📂 Estrutura de Diretórios

```bash
.
├── config/           # Configurações gerais
├── data/             # Dados brutos, processados e datasets finais
├── experiments/      # (controlado via config/)
├── models/           # Modelos treinados
├── notebooks/        # Análises exploratórias
├── runs/             # Resultados automáticos dos studies
├── src/              # Código-fonte completo do pipeline
├── tests/            # Testes
├── README.md         # (este documento)
└── environment.yml   # Ambiente Conda
