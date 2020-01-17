from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:,:2]
Y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)

# print("\nAccuracy is ( in % ):",metrics.accuracy_score(y_test,y_pred)*100)

plt.scatter(X[:,0],X[:,1],c=Y)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()