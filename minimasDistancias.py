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

# Filtrar para incluir solo las clases 0 y 1
selected_classes = [0, 1]
selected_indices = [i for i in range(len(y)) if y[i] in selected_classes]

X_two_classes = X[selected_indices]
y_two_classes = y[selected_indices]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_two_classes)

# Calcular la matriz de covarianza
cov_matrix = np.cov(X_scaled, rowvar=False)

# Calcular los valores propios y vectores propios
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordenar los valores propios y vectores propios en orden descendente
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Seleccionar el número de componentes principales
num_components = 2

top_eigenvectors = eigenvectors[:, :num_components]

# Proyectar los datos originales en el nuevo espacio de características
X_pca = X_scaled.dot(top_eigenvectors)


# Crear un DataFrame para los componentes principales
pca_df = pd.DataFrame(data=X_pca, columns=['Componente Principal 1', 'Componente Principal 2'])



pca_df['Target'] = y_two_classes

# Visualizar los resultados
plt.figure(figsize=(10, 6))
targets = np.unique(y_two_classes)
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indices_to_keep = pca_df['Target'] == target
    plt.scatter(pca_df.loc[indices_to_keep, 'Componente Principal 1'],
                pca_df.loc[indices_to_keep, 'Componente Principal 2'],
                c=color, label=target)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(targets)
plt.title('PCA en el conjunto de datos Iris')
plt.show()

# Las dos características más relevantes según PCA, con dos clases
#sacar centroides
centroide1= sacarCentroides(X_pca[y_two_classes == 0])
centroide2= sacarCentroides(X_pca[y_two_classes == 1])

#sacar recta divisira
x_vals = np.linspace(min(X_pca[:, 0]), max(X_pca[:, 0]), 10)
# x_vals = np.linspace(-5,5,100)
y_vals = ecuacion_lineal(x_vals)

# Visualizar los resultados para dos clases
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Recta', color='red')
targets_2classes = np.unique(y_two_classes)
colors_2classes = ['r', 'g']
for target, color in zip(targets_2classes, colors_2classes):
    indices_to_keep = y_two_classes == target
    plt.scatter(X_pca[indices_to_keep, 0], X_pca[indices_to_keep, 1], c=color, label=f'Clase {target}')
plt.scatter(centroide1[0], centroide1[1], marker='s', color='blue', label='Centroide 0')
plt.scatter(centroide2[0], centroide2[1], marker='s', color='purple', label='Centroide 1')
plt.xlabel('PCA Componente 1')
plt.ylabel('PCA Componente 2')
plt.grid("on")
plt.title('PCA en el conjunto de datos Iris (Dos clases)')
plt.legend()
plt.show()

#predict
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2test)

# Calcular la matriz de covarianza
cov_matrix = np.cov(X_scaled, rowvar=False)

# Calcular los valores propios y vectores propios
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordenar los valores propios y vectores propios en orden descendente
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Seleccionar el número de componentes principales
num_components = 2

top_eigenvectors = eigenvectors[:, :num_components]

# Proyectar los datos originales en el nuevo espacio de características
X_pca = X_scaled.dot(top_eigenvectors)

#evaluacion de predicciones (test)
predictVals = predict(X_pca[:,0],X_pca[:,1])

for i in range(len(predictVals)):
  if predictVals[i] > 0:
    print(f" el valor {X_pca[i]} pertenece a la clase 1")
  else:
    print(f" el valor {X_pca[i]} pertenece a la clase 2")

predictVals = predict(X_pca[:, 0], X_pca[:, 1])
# Transformar las predicciones a clases (0 o 1)
predicted_classes = np.where(predictVals > 0, 1, 0)
conf_matrix = confusion_matrix(y2test, predicted_classes)
accuracy = accuracy_score(y2test, predicted_classes)
print("Matriz de Confusión:")
print(conf_matrix)
print("\nAccuracy:", accuracy)

def ecuacion_lineal(x):
    restad1= ((centroide1[0]**2) + (centroide1[1]**2))/2
    restad2= ((centroide2[0]**2) + (centroide2[1]**2))/2
    restad3= ((centroide3[0]**2) + (centroide3[1]**2))/2
    dc1 = (centroide1[0]-centroide2[0])
    dc2 = (centroide1[1]-centroide2[1])
    dc3 = (centroide3[1]-centroide3[1])
    r1 = (restad1-restad2)
    r2 = (restad1-restad3)
    return -(dc1/dc2)*x - (r1/dc2), -(dc1/dc3)*x - (r2/dc3)

#Las dos características más relevantes según t-SNE, con tres clases.
X3clases = X_train[y_train != 3]
y3clases = y_train[y_train != 3]


scaler = StandardScaler()
X_scaled_3classes = scaler.fit_transform(X3clases)

tsne_3classes = TSNE(n_components=2, random_state=42)
X_tsne_3classes = tsne_3classes.fit_transform(X_scaled_3classes)

#fit con 3 centroides
centroide1= np.array(sacarCentroides(X_tsne_3classes[y3clases == 0]))
centroide2= np.array(sacarCentroides(X_tsne_3classes[y3clases == 1]))
centroide3= np.array(sacarCentroides(X_tsne_3classes[y3clases == 2]))

x_vals = np.linspace(min(X_tsne_3classes[:, 0]), max(X_tsne_3classes[:, 0]), 100)
y_vals = ecuacion_lineal(x_vals)

# Visualizar los resultados para dos clases
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals[0], label='Recta', color='red')
plt.plot(x_vals, y_vals[1], label='Recta', color='red')
targets_3classes = np.unique(y3clases)
colors_3classes = ['r', 'g','b']
for target, color in zip(targets_3classes, colors_3classes):
    indices_to_keep = y3clases == target
    plt.scatter(X_tsne_3classes[indices_to_keep, 0], X_tsne_3classes[indices_to_keep, 1], c=color, label=f'Clase {target}')
plt.scatter(centroide1[0], centroide1[1], marker='s', color='blue', label='Centroide 0')
plt.scatter(centroide2[0], centroide2[1], marker='s', color='purple', label='Centroide 1')
plt.scatter(centroide3[0], centroide3[1], marker='s', color='orange', label='Centroide 2')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.grid("on")
plt.title('t-SNE en el conjunto de datos Iris (Tres clases)')
plt.legend()
plt.show()

def euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

X3clasesTest = X_test[y_test != 3]
y3clasesTest = y_test[y_test != 3]


scaler = StandardScaler()
X_scaled_3classes = scaler.fit_transform(X3clasesTest)

tsne_3classes = TSNE(n_components=2, random_state=42, perplexity=1)
X_tsne_3classes = tsne_3classes.fit_transform(X_scaled_3classes)
centroides = np.array([[centroide1],[centroide2],[centroide3]])


predicted_classes_3classes = []
for punto in X_tsne_3classes:
    distancias = [euclidiana(punto, centroide) for centroide in centroides]
    clase_predicha = np.argmin(distancias)
    predicted_classes_3classes.append(clase_predicha)
    print(f"El punto {punto} pertenece a la clase {clase_predicha}")

#Matriz de confusión y accuracy
predicted_classes_3classes = np.array(predicted_classes_3classes)
conf_matrix_3classes = confusion_matrix(y3clasesTest, predicted_classes_3classes)
accuracy_3classes = accuracy_score(y3clasesTest, predicted_classes_3classes)
print("Matriz de Confusión:")
print(conf_matrix_3classes)
print("\nAccuracy:", accuracy_3classes)