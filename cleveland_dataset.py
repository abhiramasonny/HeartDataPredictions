import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import tensorflow as tf
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
cleveland = pd.read_csv(url, names=names)

print(cleveland.shape)
print(cleveland.loc[1])

cleveland.loc[280:]
data = cleveland[~cleveland.isin(['?'])]
data = data.dropna(axis=0)

#rando data i dont like!
data = data.drop(columns=['restecg', 'oldpeak', 'ca', 'slope'])

print(data.shape)
print(data.dtypes)

data = data.apply(pd.to_numeric)
data.hist(figsize=(12, 12))
plt.show()

X = data.drop('class', axis=1).values
y = data['class'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

Y_train = tf.keras.utils.to_categorical(y_train, num_classes=None)
Y_test = tf.keras.utils.to_categorical(y_test, num_classes=None)
print(Y_train.shape)
print(Y_train[:10])

def create_model():
    model = Sequential()
    model.add(Dense(8, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()
Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1
print(Y_train_binary[:20])

def create_binary_model():
    model = Sequential()
    model.add(Dense(8, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

binary_model = create_binary_model()
print(binary_model.summary())
binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)

categorical_pred = np.argmax(model.predict(X_test), axis=1)
print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

binary_pred = np.round(binary_model.predict(X_test)).astype(int)
print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))

model.save("models/categorical_model_dropped.h5")
binary_model.save("models/binary_model_dropped.h5")
