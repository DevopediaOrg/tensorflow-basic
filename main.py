import tensorflow as tf

h = tf.constant("Hello", name='Hello')
w = tf.constant("World", name='World')

hw = h + w

print(hw)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run([h, w, hw]))