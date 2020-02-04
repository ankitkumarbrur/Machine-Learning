import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("headbrain.csv")

print(dataset.shape)

X = dataset["Head Size(cm^3)"].values
print(type(X))
# print(X)
# print(X.T)