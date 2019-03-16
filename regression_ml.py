import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv('./Advertising.csv')

X = tf.placeholder("float32", name="X")
Y = tf.placeholder("float32", name="Y")

with tf.name_scope("Model"):
    w = tf.Variable(0.0, )

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run([slope, intercept]))

plt.scatter(df.TV.values, df.Sales.values)
plt.show()