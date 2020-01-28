import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("headbrain.csv")

print(dataset.shape)

X = dataset["Head Size(cm^3)"].values
Y = dataset["Brain Weight(grams)"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X_mean = np.mean(X_train)
Y_mean = np.mean(Y_train)

n = len(X_train)

num = np.sum((X_train - X_mean)*(Y_train - Y_mean))
den = np.sum((X - X_mean)**2)

b1 = num/(den)

b0 = Y_mean - b1*X_mean

print(b1,b0)

X_max = np.max(X_train) + 100
X_min = np.min(X_train) - 100

x_tr = np.linspace(X_min,X_max,1000)

y_tr = b0 + b1*x_tr

plt.plot(x_tr,y_tr,label="line",color="#ff0000")
plt.scatter(X_train,Y_train,label="points")

plt.legend()  

def acc(x_actual, y_actual):
    y_pred = b0 + b1 * x_actual 
    n = len(x_actual)
    rmse = np.sqrt((np.sum((y_actual - y_pred) ** 2) / n)) 
    rss = np.sum((y_actual - y_pred) ** 2)
    tss = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (rss / tss)

    return rmse, r2


train_rmse, train_r2 = acc(X_train, Y_train)
test_rmse, test_r2 = acc(X_test, Y_test)

print(f'Training Set\nRMSE = {train_rmse}\nR2 = {train_r2}')
print(f'Testing Set\nRMSE = {test_rmse}\nR2 = {test_r2}')

plt.show()