import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("headbrain.csv")

print(dataset.shape)

X = dataset["Head Size(cm^3)"].values
X= (X-np.min(X))/(np.max(X)-np.min(X))
# print(X)
# print(X.T)
Y = dataset["Brain Weight(grams)"].values
Y= (Y-np.min(Y))/(np.max(Y)-np.min(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# print(X_train)
m = len(X_train)
# print(np.shape(X_train))
ones = np.ones(m)
X_train= X_train.reshape(m,1)
ones=ones.reshape(m,1)
# print(np.shape(ones))
X_train = np.concatenate((ones,X_train),axis=1)
X_train=X_train
# print(np.shape(X_train))
# print(X_train)
n = len(X_test)
ones=np.ones(n).reshape(n,1)
X_test=X_test.reshape(n,1)
X_test = np.concatenate((ones,X_test),axis=1)
# print(np.shape(X_test))
# def  cal_cost(theta,X,y):
#     '''
    
#     Calculates the cost for given X and Y. The following shows and example of a single dimensional X
#     theta = Vector of thetas 
#     X     = Row of X's np.zeros((2,j))
#     y     = Actual y's np.zeros((2,1))

#     where:
#         j is the no of features
#     '''
    
#     m = len(y)
    
#     predictions = X.dot(theta)
#     cost = (1/2*m) * np.sum(np.square(predictions-y))
#     return cost

def gradient_descent(X_train, X_test, Y_train, Y_test,theta,learning_rate=0.01,iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    # print(np.shape(X_test))
    m = len(Y_train)
    n=len(Y_test)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    rmse_train=np.zeros(iterations)
    rmse_test=np.zeros(iterations)

    for it in range(iterations):
        prediction_train = X_train.dot(theta)
        prediction_test = X_test.dot(theta)
        difference_train=prediction_train-Y_train
        difference_test=prediction_test-Y_test
        rmse_train[it] = np.sqrt((np.sum((difference_train) ** 2) / m))
        rmse_test[it] = np.sqrt((np.sum((difference_test) ** 2) /n))

        theta[0] = theta[0] -((2/m)*learning_rate*(np.sum(X_train.T[1].dot(difference_train))))
        theta[1] = theta[1] -((2/m)*learning_rate*(np.sum(difference_train)))
    return theta, rmse_train,rmse_test
        
lr = 0.01
n_iter = 1000

theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1)),X]
# print(np.shape(X_train))
# print(np.shape(X_test))
theta,rmse_train,rmse_test = gradient_descent(X_train,X_test,Y_train,Y_test,theta,lr,n_iter)
# plt.scatter(n_iter,rmse_train, color="red")
# plt.scatter(n_iter,rmse_test,color="green")
# plt.show()
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
# print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))