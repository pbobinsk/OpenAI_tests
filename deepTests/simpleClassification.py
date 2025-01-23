from plots import plot_decision_regions, show_plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Generowanie danych
X, y = make_classification(n_features=2, n_classes=3, n_redundant=0, random_state=42, n_clusters_per_class=1)
# Klasyfikator drzewo decyzyjne
dtclf = DecisionTreeClassifier().fit(X, y)

plot_decision_regions(X, y, dtclf)
show_plot(['Cecha 1','Cecha 2'],'Zbiór losowy - Drzewo decyzyjne')
plt.close('all')

# Klasyfikacja na zbiorze Iris

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Etykiety klas:', np.unique(y))

# Drzewo decyzyjne
dtclf = DecisionTreeClassifier().fit(X, y)
plot_decision_regions(X, y, dtclf,test_idx=range(105, 150))
show_plot(['Cecha 1','Cecha 2'],'Zbiór Iris - Drzewo decyzyjne')
plt.close('all')

# Perceptron

# Dzielimy dane na 70% przykładów uczących i 30% przykładów testowych:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

# Standaryzacja cech:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Uczenie Perceptronu

from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Nieprawidłowo sklasyfikowane przykłady: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Dokładność: %.3f' % accuracy_score(y_test, y_pred))

print('Dokładność: %.3f' % ppn.score(X_test_std, y_test))

# Trenowanie modelu perceptronu za pomocą standaryzowanych danych uczących:

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
show_plot(['Długość płatka [standaryzowana]','Szerokość płatka [standaryzowana]'],
          'Zbiór Iris - Perceptron')
