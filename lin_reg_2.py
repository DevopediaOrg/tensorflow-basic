import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

graph_dir = "./graphs"

df = pd.read_csv('data/Advertising.csv')

X = tf.placeholder("float32", name="X")
Y = tf.placeholder("float32", name = "Y")

with tf.name_scope("Model"):
    w = tf.Variable(0.0, name="b1", dtype="float32")
    b = tf.Variable(0.0, name="b0", dtype="float32")
    y_model = tf.multiply(X, w) + b

with tf.name_scope("CostFunction"):
    cost = tf.pow(Y-y_model, 2, name="cost")/2

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    cost_op = tf.summary.scalar("loss", cost)
    w_op = tf.summary.histogram("weight", w)
    merged = tf.summary.merge_all()
    sess.run(init)
    writer = tf.summary.FileWriter(graph_dir, sess.graph)
    final_w = 0.0
    final_b = 0.0
    tv = df.TV.values / np.max(df.TV.values)
    sales = df.Sales.values / np.max(df.Sales.values)
    for i in range(200):
        for (j, (x, y)) in enumerate(zip(tv, sales)):
            feed_dict = {X:x, Y:y}
            summary, train, final_w, final_b = sess.run([merged, train_op, w, b], feed_dict=feed_dict)
            idx = (i * len(tv)) + j
            writer.add_summary(summary, idx)
        if i%10 == 0:
            plt.scatter(tv, sales)
            plt.plot(tv, final_b + (final_w * tv), color='red')
            plt.savefig(str(i))
            plt.close()
