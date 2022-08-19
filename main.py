 #import libraries
import os

import matplotlib
import numpy as np #making arrays
import pandas as pd #for data management
import matplotlib.pyplot as plt #for data visualization

dataset_train = pd.read_csv("/Users/hoangcongtri/Desktop/CS/Python/GOOGStock-Price-Prediction/Google_Stock_Price_Train.csv")
dataset_train.head()
#Using Google's stock open price to train the model
training_set= dataset_train.iloc[:, 1:2].values
print(training_set)
print(training_set.shape)

#Normalizing the dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set

#Creating data structure with 60 timesteps and 1 output
X_train = []
Y_train = []
for i in range(60, 1258):
   X_train.append(scaled_training_set[i-60:i, 0])
   Y_train.append(scaled_training_set[i:0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



