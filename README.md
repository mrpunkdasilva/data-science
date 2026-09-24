# Data Science and Machine Learning Vault

Repositório contendo labs, atividades (assignments) e capstone do curso CSDS-352.

## Table of Contents

### Labs

| Lab | Arquivo |
|-----|---------|
| Lab 1.1 - Wisconsin Cancer Analysis | [wisconsin-cancer-analysis.ipynb](labs/solutions/lab-1/wisconsin-cancer-analysis.ipynb) |
| Lab 1.2 - Auto MPG Analysis | [auto-mpg-analysis.ipynb](labs/solutions/lab-1/auto-mpg-analysis.ipynb) |
| Lab 2.1 - Wisconsin Cancer Classification | [2.lab.1.wisconsin-cancer-classification.ipynb](labs/solutions/lab-2/2.lab.1.wisconsin-cancer-classification.ipynb) |
| Lab 2.2 - Auto MPG Regression | [2.lab.2.auto-mpg-regression.ipynb](labs/solutions/lab-2/2.lab.2.auto-mpg-regression.ipynb) |
| Lab 3.1 - Wisconsin Cancer Classification NN | [3.lab.1.wisconsin-cancer-classification-nn.ipynb](labs/solutions/lab-3/3.lab.1.wisconsin-cancer-classification-nn.ipynb) |
| Lab 3.2 - Auto MPG Regression NN | [3.lab.2.auto-mpg-regression-nn.ipynb](labs/solutions/lab-3/3.lab.2.auto-mpg-regression-nn.ipynb) |
| Lab 4 - Transfer Learning Hypothesis | [4.labtransfer-learning-hypothesis.ipynb](labs/solutions/lab-4/4.labtransfer-learning-hypothesis.ipynb) |

### Glimpses

| Visão Geral | Arquivo |
|-------------|---------|
| Ativações | [activations.ipynb](labs/glimpses/activations.ipynb) |
| XOR | [xor.ipynb](labs/glimpses/xor.ipynb) |
| XOR Neural Network | [xor_nn.ipynb](labs/glimpses/xor_nn.ipynb) |

### Atividades (Assignments)

| Atividade | Arquivo |
|-----------|---------|
| Tarefa 1 | [solution.ipynb](assignments/assignment-1/solution.ipynb) |
| Tarefa 2 | [2.4.1.polynomial-regression-sgd.ipynb](assignments/assignment-2/2.4.1.polynomial-regression-sgd.ipynb) |
| Tarefa 3 | [3.4.1.minimal-xor-network.ipynb](assignments/assignment-3/3.4.1.minimal-xor-network.ipynb) |

### Capstone

| Capstone | Pasta |
|----------|-------|
| Projeto Final | [capstone/](capstone/) |

### Templates

| Template | Arquivo |
|----------|---------|
| 1.4.1 - BMI Exploration | [1.4.1.bmi-exploration.ipynb](templates/1.4.1.bmi-exploration.ipynb) |
| 1 Lab 1 - Wisconsin Cancer Analysis | [1.lab.1.wisconsin-cancer-analysis.ipynb](templates/1.lab.1.wisconsin-cancer-analysis.ipynb) |
| 1 Lab 2 - Auto MPG Analysis | [1.lab.2.auto-mpg-analysis.ipynb](templates/1.lab.2.auto-mpg-analysis.ipynb) |
| 2.4.1 - Polynomial Regression SGD | [2.4.1.polynomial-regression-sgd.ipynb](templates/2.4.1.polynomial-regression-sgd.ipynb) |
| 2 Lab 1 - Wisconsin Cancer Classification | [2.lab.1.wisconsin-cancer-classification.ipynb](templates/2.lab.1.wisconsin-cancer-classification.ipynb) |
| 2 Lab 2 - Auto MPG Regression | [2.lab.2.auto-mpg-regression.ipynb](templates/2.lab.2.auto-mpg-regression.ipynb) |
| 3.4.1 - Minimal XOR Network | [3.4.1.minimal-xor-network.ipynb](templates/3.4.1.minimal-xor-network.ipynb) |
| 3 Lab 1 - Wisconsin Cancer Classification NN | [3.lab.1.wisconsin-cancer-classification-nn.ipynb](templates/3.lab.1.wisconsin-cancer-classification-nn.ipynb) |
| 3 Lab 2 - Auto MPG Regression NN | [3.lab.2.auto-mpg-regression-nn.ipynb](templates/3.lab.2.auto-mpg-regression-nn.ipynb) |

## Como rodar

1. Entre na pasta `labs/`:
   ```bash
   cd labs
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -e .
   ```

4. Para rodar os notebooks, inicie o Jupyter:
   ```bash
   jupyter lab
   ```

Ou execute diretamente com:
```bash
ipython
```

## Dependências

- Python >= 3.12
- matplotlib, numpy, pandas, scikit-learn, torch, torchvision, seaborn
- ipykernel, ipympl, rich
