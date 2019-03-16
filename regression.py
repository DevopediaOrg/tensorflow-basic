import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv('./Advertising.csv')

X = tf.constant(df.TV.values)
Xa = tf.reduce_mean(X)

Y = tf.constant(df.Sales.values)
Ya = tf.reduce_mean(Y)

numerator = tf.reduce_sum((Y - Ya) * (X - Xa))
denominator = tf.reduce_sum(tf.pow((X - Xa), 2))
slope = numerator / denominator
intercept = Ya - (slope * Xa)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run([slope, intercept]))

plt.scatter(df.TV.values, df.Sales.values)
plt.show()
input()