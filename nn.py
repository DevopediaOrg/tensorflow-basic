import pandas as pd
import numpy as np

def read_data():
  df = pd.read_csv('./Advertising.csv')
  X = df[['TV', 'Radio']].as_matrix()
  X = X / np.max(X)
  Y = df[['Sales']].as_matrix()
  Y = Y / np.max(Y)
  return X, Y

num_neurons = 3

def init_weights(X, Y):
  w1 = np.random.rand(X.shape[1], num_neurons)
  w2 = np.random.rand(num_neurons, Y.shape[1])
  return w1, w2

def activation(z):
  return 1 / (1 + np.exp(-z))

def forward_propagate(X, w1, w2):
  z1 = np.matmul(X, w1)
  a1 = activation(z1)
  z2 = np.matmul(a1, w2)
  y_hat = activation(z2)
  return y_hat

def cost(y_hat, y):
  return 0.5*np.sum((y_hat - y)**2)

X, Y = read_data()
w1, w2 = init_weights(X, Y)
y_hat = forward_propagate(X, w1, w2)
c = cost(y_hat, Y)
print(c)
# print(X.shape, Y.shape)