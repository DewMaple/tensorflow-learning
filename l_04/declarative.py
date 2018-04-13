import tensorflow as tf

W = tf.get_variable('w', shape=1, dtype=tf.float32, initializer=tf.constant_initializer([0.3]))
b = tf.get_variable('b', shape=1, dtype=tf.float32, initializer=tf.constant_initializer([-0.3]))

x = tf.placeholder(tf.float32)
line_model = W * x + b
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(line_model - y))

train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
loss_ = 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, l = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
        loss_ = min(loss_, l)
    curr_W, curr_b = W.eval(), b.eval()

    print("W: {}, b: {}, loss: {}".format(curr_W, curr_b, loss_))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
    loss_ = min(loss_, l)

curr_W, curr_b = sess.run([W, b], feed_dict={x: x_train, y: y_train})
print("W: {}, b: {}, loss: {}".format(curr_W, curr_b, loss_))
sess.close()
