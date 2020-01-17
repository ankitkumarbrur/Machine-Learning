from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.95, random_state=1)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("\nAccuracy is ( in % ):", metrics.accuracy_score(y_test, y_pred)*100)

# print(X)

# After Normalization

x_min = X[:, 0].min()
x_max = X[:, 0].max()

X[:, 0] = (X[:, 0]-x_min)/(x_max-x_min)

x_min = X[:, 1].min()
x_max = X[:, 1].max()

X[:, 1] = (X[:, 1]-x_min)/(x_max-x_min)

x_min = X[:, 2].min()
x_max = X[:, 2].max()

X[:, 2] = (X[:, 2]-x_min)/(x_max-x_min)

x_min = X[:, 3].min()
x_max = X[:, 3].max()

X[:, 3] = (X[:, 3]-x_min)/(x_max-x_min)


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.95, random_state=1)
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("\nAccuracy is ( in % ):", metrics.accuracy_score(y_test, y_pred)*100)

# print(X)
