import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import os, ssl, time

data = pd.read_csv("labels.csv")["labels"]
print(pd.Series(data).value_counts())

X = np.load('image.npz')['arr_0']
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, data, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

clf = LogisticRegression(solver= "saga", multi_class= "multinomial").fit(X_train_scaled, Y_train)
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy * 100)