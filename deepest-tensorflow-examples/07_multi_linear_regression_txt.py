import tensorflow as tf
import numpy as np


xy = np.loadtxt('multi_linear_regression.txt', unpack = True, dtype = 'float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1,3],-1.0,1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W,X)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data,  Y:y_data}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X:[[1],[1],[1]]}))
print(sess.run(hypothesis, feed_dict={X:[[1],[2],[3]]}))

