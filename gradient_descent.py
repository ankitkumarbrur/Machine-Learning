import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def cost_func(theta, X, y):
    m = len(X)
    pred = X.dot(theta)
    cost = (1/m) * np.sum(np.square(pred - y))
    return cost**(1/2)

def gradientDescent(X_train, X_test, y_train, y_test, learning_rate=0.1, iteration=1000):
    m = len(y_train)
    theta = np.zeros(X_train.shape[1]).T
    rmse_train = np.empty(iteration)
    rmse_test = np.empty(iteration)
    it = np.arange(iteration)
    theta_history = np.zeros(X_train.shape[1]).T
    theta_history = np.reshape(theta_history,(1,2))

    for i in range(iteration):
        pred = np.dot(X_train, theta)
        theta = theta - (1 / m) * learning_rate * (X_train.T.dot((pred - y_train)))
        rmse_train[i] = cost_func(theta, X_train, y_train)
        rmse_test[i] = cost_func(theta, X_test, y_test)
        theta_history = np.append(theta_history,np.reshape(theta,(1,2)),axis=0)

    plt.plot(it, rmse_train, c='red', label='Training RMSE')
    plt.plot(it, rmse_test, c='green', label='Testing RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    print('\n\nTheta0:          {:0.3f}\nTheta1:          {:0.3f}'.format(theta[0],theta[1]))
    plt.show()
    print(theta_history)
    theta0_history = theta_history.T[0,:]
    theta1_history = theta_history.T[1,:]
    it = np.arange(iteration+1)
    # plt.cla()
    plt.plot(it, theta0_history, c='red', label='theta0')
    plt.plot(it, theta1_history, c='green', label='theta1')
    plt.xlabel('Iteration')
    plt.ylabel('Theta')
    plt.legend()
    plt.show()
    return theta, rmse_train, rmse_test


database = pd.read_csv('headbrain.csv')

scaler = MinMaxScaler()
database = scaler.fit_transform(database)

X = database[:, 2]
y = database[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = np.c_[np.ones((len(X_train), 1), dtype='int'), X_train]
X_test = np.c_[np.ones((len(X_test), 1), dtype='int'), X_test]


theta, rmse_train, rmse_test = gradientDescent(X_train, X_test, y_train, y_test, 0.01, 20000)

y_pred = theta[0] + theta[1] * X_train
plt.plot(X_train, y_pred, color='r', label='Gradient Descent')
X_train = X_train[:,1]
plt.scatter(X_train, y_train)    
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()