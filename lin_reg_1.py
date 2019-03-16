import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./Advertising.csv')

with tf.name_scope("Input"):
    X = tf.constant(df['TV'])
    Y = tf.constant(df['Sales'])

with tf.name_scope("MeanCalculation"):
    Ya = tf.reduce_mean(Y)
    Xa = tf.reduce_mean(X)

with tf.name_scope("Slope"):
    numerator = tf.reduce_sum((Y - Ya) * (X - Xa))
    denomenator = tf.reduce_sum(tf.pow((X - Xa), 2))
    slope = numerator / denomenator

with tf.name_scope("Interccept"):
    intercept = Ya - (slope * Xa)

with tf.Session() as sess:
    b1 = sess.run(slope)
    b0 = sess.run(intercept)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # plt_x = df['TV'].values
    # plt_y = df['Sales'].values
    # plt.scatter(plt_x, plt_y)
    # plt.plot(plt_x, b0 + (plt_x * b1), color='red')
    # plt.show()
