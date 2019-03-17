import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_dataset():
  df = pd.read_csv('./data/Advertising.csv')
  x = df[['TV']].as_matrix()
  y = df['Sales'].as_matrix()
  y = y.reshape(len(y), 1)
  x = MinMaxScaler().fit_transform(x)
  y = MinMaxScaler().fit_transform(y)
  return x, y

def linear_regression():
  x = tf.placeholder(tf.float32, shape=(200, 1), name='x')
  y = tf.placeholder(tf.float32, shape=(200, 1), name='y')

  with tf.variable_scope('lreg') as scope:
    w = tf.Variable(np.random.normal(), name='W')
    b = tf.Variable(np.random.normal(), name='b')
    
    y_pred = tf.add(tf.multiply(w, x), b)

    loss = tf.reduce_mean(tf.square(y_pred - y))

  return x, y, y_pred, loss

def run():
  x_batch, y_batch = generate_dataset()
  x, y, y_pred, loss = linear_regression()

  optimizer = tf.train.GradientDescentOptimizer(0.1)
  train_op = optimizer.minimize(loss)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    feed_dict = {x: x_batch, y: y_batch}
    
    for i in range(30):
      session.run(train_op, feed_dict)
      print(i, "loss:", loss.eval(feed_dict))

    print('Predicting')
    y_pred_batch = session.run(y_pred, {x : x_batch})

if __name__ == "__main__":
  run()
