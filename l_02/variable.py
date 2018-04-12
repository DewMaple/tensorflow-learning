import tensorflow as tf

#
# a = tf.constant([2, 2], name='a')
# b = tf.constant([[0, 1], [2, 3]], name='b')
#
# x = tf.multiply(a, b, name='X')
# writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
# with tf.Session() as sess:
#     print(sess.run(x))
# writer.close()


# my_const = tf.constant([1.0, 2.0], name="my_const")
# print(tf.get_default_graph().as_graph_def())


# s = tf.Variable(2, name="scalar")
# m = tf.Variable([[0, 1], [2, 3]], name="matrix")
#
# W = tf.Variable(tf.zeros([784,10]))
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(s))
    print(session.run(W))
print(m)
