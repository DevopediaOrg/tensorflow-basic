import tensorflow as tf

h = tf.constant('hello', name='h')
w = tf.constant('world', name='w')
hw = h+w

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(w))
