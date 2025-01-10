
from sklearn import datasets
iris = datasets.load_iris() 
data = iris.data 
labels = iris.target

# print(data)
# print(labels)

from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(random_state=1) 
mlp.fit(data, labels)

pred = mlp.predict(data)
probabilities = mlp.predict_proba(data)

from sklearn.metrics import accuracy_score 
print(f'Dokładność: {accuracy_score(labels, pred)}')
#print(f'Prawdopodobieństwo: {probabilities}')

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=1)  
scaler = StandardScaler() 
scaler.fit(data) 
data_train_std = scaler.transform(data_train) 
data_test_std = scaler.transform(data_test)  
data_train_std = data_train 
data_test_std = data_test

mlp.fit(data_train, labels_train)
pred = mlp.predict(data_test)

print('Błędnie sklasyfikowane próbki: %d' % (labels_test != pred).sum())


print(f'Dokładność: {accuracy_score(labels_test, pred)}')

# import numpy
# from matplotlib.colors import ListedColormap 
# import matplotlib.pyplot as plt
# markers = ('s', '*', '^') 
# colors = ('blue', 'green', 'red') 
# cmap = ListedColormap(colors)   
# x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1 
# y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1 
# resolution = 0.01  
# x, y = numpy.meshgrid(numpy.arange(x_min, x_max, resolution), numpy.arange(y_min, y_max, resolution)) 
# Z = mlp.predict(numpy.array([x.ravel(), y.ravel()]).T) 
# Z = Z.reshape(x.shape)  
# plt.pcolormesh(x, y, Z, cmap=cmap) 
# plt.xlim(x.min(), x.max()) 
# plt.ylim(y.min(), y.max())  
# # rysowanie wykresu danych 
# classes = ["setosa", "versicolor", "verginica"] 
# for index, cl in enumerate(numpy.unique(labels)):     
# plt.scatter(data[labels == cl, 0], data[labels == cl, 1], c=cmap(index), marker=markers[index], s=50, label=classes[index])     
# plt.xlabel('petal length') 
# plt.ylabel('sepal length') 
# plt.legend(loc='upper left') 
# plt.show() 

# mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(200, 100,))

# mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(200, 100,), activation = "logistic")
