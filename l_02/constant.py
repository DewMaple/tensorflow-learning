import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    a = tf.constant(2)
    b = tf.constant(3)
    x = tf.add(a, b)
    with tf.Session() as session:
        print(session.run(x))
