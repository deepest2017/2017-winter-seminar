import tensorflow as tf
import numpy as np

xy = np.loadtxt('softmax_regression.txt', unpack = True, dtype = 'float32')
x_data = np.transpose(xy[0:-3])
y_data = np.transpose(xy[-3:])

W = tf.Variable(tf.zeros([3,3]))

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

hypothesis = tf.nn.softmax(tf.matmul(X,W))

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices = 1))

a = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data,  Y:y_data}), sess.run(W))

print('---------------------------------')

a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
print(a, sess.run(tf.argmax(a,1)))

b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
print(b, sess.run(tf.argmax(b,1)))

c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
print(c, sess.run(tf.argmax(c,1)))

all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print(all, sess.run(tf.argmax(all,1)))


