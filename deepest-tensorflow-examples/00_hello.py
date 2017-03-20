import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')

sess = tf.Session()

print (hello)

print (sess.run(hello))

