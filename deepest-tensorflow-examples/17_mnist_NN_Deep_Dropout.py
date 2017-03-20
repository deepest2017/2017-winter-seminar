import tensorflow as tf

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[784, 256], initializer = xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer = xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer = xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer = xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer = xavier_init(256, 10))

b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([256]))
b4 = tf.Variable(tf.zeros([256]))
b5 = tf.Variable(tf.zeros([10]))

dropout_rate = tf.placeholder("float")

L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1, dropout_rate)

L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2, dropout_rate)

L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
L3 = tf.nn.dropout(L3, dropout_rate)

L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
L4 = tf.nn.dropout(L4, dropout_rate)

hypothesis = tf.matmul(L4,W5)+b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

a = tf.Variable(0.001)
optimizer = tf.train.AdamOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 100
display_step = 1
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:0.7})
        avg_cost += sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})/total_batch

    if epoch % display_step == 0:
        print ("Epoch:", '%04d' %(epoch+1), "cost:", "{:0.9f}".format(avg_cost))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, dropout_rate:1.0}))

