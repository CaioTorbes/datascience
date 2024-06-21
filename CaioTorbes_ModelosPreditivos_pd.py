import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px

heart_disease = pd.read_csv('heart_disease_dataset.csv')

#Escolha das variáveis
variables = ['Age', 'Cholesterol', 'Exercise Hours', 'Stress Level']
x = heart_disease[variables]
y = heart_disease['Heart Disease']

description = x.describe()

means = description.loc['mean']
stds = description.loc['std']

# Exibir médias e desvios padrões
print("Médias das variáveis:")
print(means)
print("\nDesvios padrões das variáveis:")
print(stds)


# GRÁFICOS PARA VISUALIZAÇÃO DOS DADOS
# Distribuição da idade dos avaliados - Box plot
fig = px.box(heart_disease, x="Age")
fig.update_traces(line_color="blue")
fig.show()

variables_graph = ['Age', 'Cholesterol', 'Exercise Hours', 'Stress Level', 'Heart Disease']
heart_disease_graph = heart_disease[variables_graph]

sns.pairplot(heart_disease_graph, hue='Heart Disease')
plt.show()


# Checar se a base está desbalanceada
class_counts = y.value_counts()
print("Distribuição das classes:")
print(class_counts)

is_balanced = class_counts.min() / class_counts.max() > 0.5
print(f"A base está balanceada? {'Sim' if is_balanced else 'Não'}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Rede Neural': MLPClassifier(max_iter=1000)
}

# Função para treinar e avaliar modelos
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    return accuracy, precision, recall, f1

# Divide os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Avaliação dos modelos
results = {}
for name, model in models.items():
    accuracy, precision, recall, f1 = train_and_evaluate(model, x_train, y_train, x_test, y_test)
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Imprimir resultados
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()

# Escolha do melhor modelo com base nas métricas
best_model = max(results, key=lambda x: results[x]['f1_score'])
print(f"O melhor modelo para operação é: {best_model}")