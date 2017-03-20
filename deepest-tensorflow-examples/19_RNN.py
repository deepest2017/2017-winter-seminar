import tensorflow as tf

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w:i for i,w in enumerate(char_rdic)}

ground_truth = [char_dic[c] for c in 'hello']

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0)

rnn_size = len(char_dic)
time_step_size = 4
batch_size = 1
output_size = 4

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)

x_split = tf.split(0, time_step_size, x_data)

output, state = tf.nn.rnn(cell = rnn_cell, inputs = x_split,
                          initial_state = initial_state)

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
# targets: list of 1D batch-sized int32 Tensors of the same length as logits
# weights: list of 1D batch-sized float-Tensors of the same length as logits
logits = tf.reshape(tf.concat(1,output), [-1, rnn_size])
targets = tf.reshape(ground_truth[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[targets],[weights])
cost = tf.reduce_sum(loss) / batch_size

a = tf.Variable(0.01)
optimizer = tf.train.RMSPropOptimizer(a, 0.9)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    result = sess.run(tf.argmax(logits, 1))
    print (result, [char_rdic[t] for t in result])

