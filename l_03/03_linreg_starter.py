""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf

import utils
from matplotlib import pyplot as plt

DATA_FILE = '../data/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
X = tf.placeholder(name='X', dtype=tf.float32)
Y = tf.placeholder(name='Y', dtype=tf.float32)
#############################
########## TO DO ############
#############################

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable(name='w', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
b = tf.get_variable(name='b', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
#############################
########## TO DO ############
#############################

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = tf.add(tf.multiply(w, X), b, name='y_predicted')
#############################
########## TO DO ############
#############################

# Step 5: use the square error as the loss function
loss = tf.square(Y_predicted - Y, name='loss')
#############################
########## TO DO ############
#############################

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
#############################
########## TO DO ############
#############################
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    #############################
    ########## TO DO ############
    #############################
    sess.run(tf.global_variables_initializer())
    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            # Execute train_op and get the value of loss.
            # Don't forget to feed in data for placeholders
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: x, Y: y})  ########## TO DO ############
            total_loss += loss_

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    #############################
    ########## TO DO ############
    #############################
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = w.eval(), b.eval()
    #############################
    ########## TO DO ############
    #############################
print('w is {}, b is {}'.format(w_out, b_out))
print('Took: %f seconds' % (time.time() - start))

# uncomment the following lines to see the plot 
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
