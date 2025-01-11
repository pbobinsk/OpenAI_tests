import matplotlib.pyplot as plt

# Przykładowe dane
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Wykres punktowy
plt.scatter(x, y, c='blue', s=100, marker='o', label='Dane')
plt.title('Podstawowy wykres punktowy')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Dane
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 1000  # Rozmiar punktów
colors = np.random.rand(50)        # Kolory punktów
# print(colors)
# Wykres punktowy
plt.scatter(x, y, s=sizes, c=colors, cmap='coolwarm', alpha=0.7, edgecolor='k', marker='*')
plt.colorbar(label='Wartość koloru')
plt.title('Punkty z różnym rozmiarem i kolorem')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generowanie siatki punktów
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Funkcja do wizualizacji
Z = np.sin(np.sqrt(X**2 + Y**2))

# Wykres wypełnionych konturów
plt.contourf(X, Y, Z, levels=1, cmap='viridis')
plt.colorbar(label='Wartość funkcji')
plt.title('Wypełnione kontury funkcji')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()

# Siatka
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Funkcja kwadratowa
Z = X**2 + Y**2  # Funkcja kwadratowa

# Wykres z ręcznym ustawieniem poziomów
plt.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8, 10, 12], cmap='coolwarm', alpha=0.8)
plt.colorbar(label='Wartość funkcji')
plt.title('Kontury z określonymi poziomami')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Generowanie danych
X, y = make_classification(n_features=2, n_classes=3, n_redundant=0, random_state=42, n_clusters_per_class=1)
clf = DecisionTreeClassifier().fit(X, y)

# Siatka punktów
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predykcja modelu na siatce
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.close('all')

print(y)


# Wykres obszarów decyzyjnych
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), alpha=0.6)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))

for idx, cl in enumerate(np.unique(y)):
        print(cl)
        print(y)
        print(y[y == cl])

        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    # c=['#FF0000', '#00FF00', '#0000FF'][idx], 
                    c=y[y==cl],
                    cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']),
                    edgecolor='k', 
                    marker=('s', '*', '^')[idx],
                    label=cl)

plt.title('Obszary decyzyjne klasyfikatora')
plt.xlabel('Cechy 1')
plt.ylabel('Cechy 2')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
