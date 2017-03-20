import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev = 0.01))
w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01))
w3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01))

p_keep_conv = tf.placeholder(tf.float32)

l1a = tf.nn.relu(tf.nn.conv2d(X, w1, [1,1,1,1], padding = 'SAME'))
l1 = tf.nn.max_pool(l1a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1,1,1,1], padding = 'SAME'))
l2 = tf.nn.max_pool(l2a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1,1,1,1], padding = 'SAME'))
l3 = tf.nn.max_pool(l3a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

w4 = tf.Variable(tf.random_normal([128*4*4, 625], stddev = 0.01))
w5 = tf.Variable(tf.random_normal([625, 10], stddev = 0.01))

p_keep_hidden = tf.placeholder(tf.float32)

l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, p_keep_hidden)

hypothesis = tf.matmul(l4, w5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

a = tf.Variable(0.001)
optimizer = tf.train.RMSPropOptimizer(a, 0.9)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 128
display_step = 1
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = trX[i*batch_size:(i+1)*batch_size], trY[i*batch_size:(i+1)*batch_size]
        sess.run(train, feed_dict={X:batch_xs, Y:batch_ys, p_keep_conv:0.8, p_keep_hidden:0.5})
        avg_cost += sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys, p_keep_conv:1.0, p_keep_hidden:1.0})/total_batch

    if epoch % display_step == 0:
        print ("Epoch:", '%04d' %(epoch+1), "cost:", "{:0.9f}".format(avg_cost), end=' ')

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:256]

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Accuracy:", sess.run(accuracy, feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0}))



