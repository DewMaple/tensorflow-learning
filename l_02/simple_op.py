import tensorflow as tf

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))  # ⇒ [[0 0][1 1]]
    print(sess.run(tf.divide(b, a)))  # ⇒ [[0. 0.5][1. 1.5]]
    print(sess.run(tf.truediv(b, a)))  # ⇒ [[0. 0.5][1. 1.5]]
    print(sess.run(tf.floordiv(b, a)))  # ⇒ [[0 0][1 1]]
    # print(sess.run(tf.realdiv(b, a)))  # ⇒  # Error: only works for real values
    print(sess.run(tf.truncatediv(b, a)))  # ⇒ [[0 0][1 1]]
    print(sess.run(tf.floor_div(b, a)))  # ⇒ [[0 0][1 1]]

t_1 = [b"apple", b"peach", b"grape"]  # 1-d arrays are treated like 1-d tensors
print(tf.zeros_like(t_1))

s = tf.Session()
print(s.run(tf.zeros_like(t_1)))
# print(s.run(tf.ones_like(t_1)))

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]

print(s.run(tf.zeros_like(t_2)))
print(s.run(tf.ones_like(t_2)))

s.close()

# W is a random 700 x 100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)
    print(W.eval())

W = tf.Variable(10)
assign_op = W.assign(100)

my_var = tf.Variable(2, name="my_var")
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

    sess.run(assign_op)
    print(W.eval())

    sess.run(my_var.initializer)
    sess.run(my_var_times_two)  # >> the value of my_var now is 4
    sess.run(my_var_times_two)
    print(my_var.eval())

    sess.run(my_var.assign_add(10))  # >> 20
    print('assign add is ', my_var.eval())
    sess.run(my_var.assign_sub(2))  # >> 18
    print('assign sub is ', my_var.eval())

W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))

print(sess1.run(W.assign_add(100)))  # >> 120
print(sess2.run(W.assign_sub(50)))

sess1.close()
sess2.close()
