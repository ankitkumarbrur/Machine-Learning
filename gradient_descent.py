import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

database = pd.read_csv('headbrain.csv')

scaler = MinMaxScaler()
database = scaler.fit_transform(database)

X = database[:, 2]
y = database[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = np.c_[np.ones((len(X_train), 1), dtype='int'), X_train]
X_test = np.c_[np.ones((len(X_test), 1), dtype='int'), X_test]
