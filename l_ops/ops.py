import tensorflow as tf

label_true = tf.constant([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 0],
    [1, 1],
    [1, 1],
    [1, 0]
])


def true_func():
    return 0


def false_func():
    return 1


label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], x[1]), true_func, false_func), label_true)

mask = tf.cast(label_filtered, dtype=tf.int32)
label_true1 = tf.boolean_mask(label_true, mask)

session = tf.Session()
label_filtered = session.run(label_filtered)
print(label_filtered)

mask = session.run(mask)
print(mask)

label_true1 = session.run(label_true1)
print(label_true1)