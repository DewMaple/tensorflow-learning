import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
    # print(sess.run(c))
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))

aa = tf.add(2, 5)
bb = tf.multiply(aa, 3)

with tf.Session() as sess:
    # compute the value of b given a is 15
    print(sess.run(bb, feed_dict={aa: 15}))  # >> 45
