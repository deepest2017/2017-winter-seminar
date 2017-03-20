import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - 0.1 * tf.reduce_mean((W*X - Y) * X)
train = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(101):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

