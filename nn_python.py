import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

num_hidden = 3

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoidPrime(z):
  return np.exp(-z)/((1+np.exp(-z))**2)

def scale(X, Y):
  return MinMaxScaler().fit_transform(X), \
    MinMaxScaler().fit_transform(Y)

def read_data():
  df = pd.read_csv('./data/Advertising.csv')
  X = df[['TV','Radio']].as_matrix()
  Y = df['Sales'].as_matrix()
  Y = Y.reshape(len(Y), 1)
  return X, Y

def init_weights(X, Y):
  return np.random.rand(X.shape[1], num_hidden), \
        np.random.rand(num_hidden, 1)

def forward_propogate(X, w1, w2):
  z1 = np.matmul(X, w1)
  a1 = sigmoid(z1)
  z2 = np.matmul(a1, w2)
  return sigmoid(z2), z2, a1, z1

def back_propogate(Y, y_hat, w1, w2, z1, z2, a1, X):
  delta2 = -np.multiply((Y-y_hat), sigmoidPrime(z2))
  dJdW2 = np.matmul(a1.T, delta2)

  delta1 = np.multiply(np.matmul(delta2, w2.T), sigmoidPrime(z1))
  dJdW1 = np.matmul(X.T, delta1)

  return dJdW1, dJdW2

def cost(y_hat, y):
  return 0.5 * sum(((y_hat - y)**2))

X, Y = read_data()
X, Y = scale(X, Y)
w1, w2 = init_weights(X, Y)
y_hat, z2, a1, z1 = forward_propogate(X, w1, w2)
print(cost(y_hat, Y))
dJdW1, dJdW2 = back_propogate(Y, y_hat, w1, w2, z1, z2, a1, X)
scalar = 0.003
w1 = w1 + (scalar * dJdW1)
w2 = w2 + (scalar * dJdW2)
y_hat, z2, a1, z1 = forward_propogate(X, w1, w2)
print(cost(y_hat, Y))