import tensorflow as tf
import numpy as np

class Model:
	def __init__ (self, vocab_size, trainable=True, input_length=1, batch_size=1, num_units=256, num_layers=1, use_lstm=False, learning_rate=0.1, gradient_clip=5.0):
		self.batch_size = batch_size
		self.input_length = input_length
		self.target_length = 1

		self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.input_length])
		self.label_data = tf.placeholder(tf.int32, [self.batch_size, self.target_length])

		if use_lstm:
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
		else:
			self.cell = tf.nn.rnn_cell.GRUCell(num_units)

		if num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * num_layers)

		self.initial_state = self.cell.zero_state(batch_size, tf.float32)
		self.step_count = tf.Variable(0, trainable=False)

		with tf.variable_scope('rnnmodel'):
			softmax_w = tf.get_variable('softmax_w', [num_units, vocab_size])
			softmax_b = tf.get_variable('softmax_b', [vocab_size])

			embedding = tf.get_variable(name='embedding', shape=[num_units, vocab_size])
			inputs = tf.split(split_dim=1, num_split=input_length, value=tf.nn.embedding_lookup(params=embedding, ids=self.input_data))
			inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

		def recursive_sample(prev, _):
			prev = tf.matmul(prev, softmax_w) + softmax_b
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, prev_symbol)

		outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=recursive_sample if not trainable else None, scope='rnnmodel')

		self.logits = tf.matmul(outputs[0], softmax_w) + softmax_b
		self.probs = tf.nn.softmax(self.logits)

		if trainable:
			# self.loss = tf.nn.seq2seq.sequence_loss_by_example(logits=[self.logits], targets=[tf.reshape(self.label_data, [-1])], weights=[tf.ones([batch_size * input_length])])
			# self.cost = tf.reduce_sum(self.loss) / batch_size / input_length

			self.loss = tf.nn.seq2seq.sequence_loss(logits=[self.logits], targets=[tf.reshape(self.label_data, [-1])], weights=[tf.ones([batch_size * input_length])])

			trainable_vars = tf.trainable_variables()
			clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), gradient_clip)

			self.trainer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.step_count)

		self.saver = tf.train.Saver()

	def step(self, session, input_data, label_data=None, training=False):
		if training:
			input_feed = {self.input_data: input_data, self.label_data:label_data}
			output_var = [self.trainer, self.loss]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.probs]

		final_output = session.run(output_var, feed_dict=input_feed)
		
		if training:
			return final_output[1]
		else:
			return final_output[0]

	def save(self, sess, file):
		save_path = self.saver.save(sess, file)
		print "Model saved at {0}".format(save_path)

	def load(self, sess, file):
		self.saver.restore(sess, file):
		print "Model restored from {0}".format(file)
