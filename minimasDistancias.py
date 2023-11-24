import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
#Aquí se hace el holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dimensiones del conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Dimensiones del conjunto de prueba:", X_test.shape, y_test.shape)


feature_names = iris.feature_names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualizar los resultados de 3 clases
plt.figure(figsize=(8, 6))
targets = np.unique(y_train)
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indices_to_keep = y_train == target
    plt.scatter(X_tsne[indices_to_keep, 0], X_tsne[indices_to_keep, 1], c=color, label=target)
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.title('t-SNE en el conjunto de datos Iris')
plt.legend()
plt.show()

# Las dos características más relevantes según t-SNE, con dos clases.
X_2classes = X_train[y_train != 2]
y_2classes = y_train[y_train != 2]


scaler = StandardScaler()
X_scaled_2classes = scaler.fit_transform(X_2classes)

tsne_2classes = TSNE(n_components=2, random_state=42)
X_tsne_2classes = tsne_2classes.fit_transform(X_scaled_2classes)


#fit
def sacarCentroides(Clase):
    return np.mean(Clase, axis=0)

centroide1= sacarCentroides(X_tsne_2classes[y_2classes == 0])
centroide2= sacarCentroides(X_tsne_2classes[y_2classes == 1])

def ecuacion_lineal(x):
    restad1= ((centroide1[0]**2) +(centroide1[1]**2))/2
    restad2= ((centroide2[0]**2) +(centroide2[1]**2))/2
    dc1= (centroide1[0]-centroide2[0])
    dc2=(centroide1[1]-centroide2[1])
    r=(restad1-restad2)
    return -(dc1/dc2)*x - (r/dc2)
def predict(x,y):
    restad1= ((centroide1[0]**2) +(centroide1[1]**2))/2
    restad2= ((centroide2[0]**2) +(centroide2[1]**2))/2
    dc1= (centroide1[0]-centroide2[0])
    dc2=(centroide1[1]-centroide2[1])
    r=(restad1-restad2)
    ecuacion = (dc1*x) + (dc2*y) - r
    return ecuacion


x_vals = np.linspace(min(X_tsne_2classes[:, 0]), max(X_tsne_2classes[:, 0]), 100)
y_vals = ecuacion_lineal(x_vals)

# Visualizar los resultados para dos clases
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Recta', color='red')
targets_2classes = np.unique(y_2classes)
colors_2classes = ['r', 'g']
for target, color in zip(targets_2classes, colors_2classes):
    indices_to_keep = y_2classes == target
    plt.scatter(X_tsne_2classes[indices_to_keep, 0], X_tsne_2classes[indices_to_keep, 1], c=color, label=f'Clase {target}')
plt.scatter(centroide1[0], centroide1[1], marker='s', color='blue', label='Centroide 0')
plt.scatter(centroide2[0], centroide2[1], marker='s', color='purple', label='Centroide 1')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.grid("on")
plt.title('t-SNE en el conjunto de datos Iris (Dos clases)')
plt.legend()
plt.show()
X2test = X_test[y_test != 2]
y2test = y_test[y_test != 2]

scaler = StandardScaler()
Xscaled2test = scaler.fit_transform(X2test)

tsne_2classes = TSNE(n_components=2, random_state=42, perplexity=1)
X2tsnetest = tsne_2classes.fit_transform(Xscaled2test)

#evaluacion de predicciones (test)
predictVals = predict(X2tsnetest[:,0],X2tsnetest[:,1])

for i in range(len(predictVals)):
    if predictVals[i] > 0:
        print(f" el valor {X2tsnetest[i]} pertenece a la clase 1")
    else:
        print(f" el valor {X2tsnetest[i]} pertenece a la clase 2")

predicted_classes_2classes = np.where(predictVals > 0, 1, 0)
conf_matrix_2classes = confusion_matrix(y2test, predicted_classes_2classes)
accuracy_2classes = accuracy_score(y2test, predicted_classes_2classes)
print("Matriz de Confusión:")
print(conf_matrix_2classes)
print("\nAccuracy:", accuracy_2classes)
