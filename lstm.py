import tensorflow as tf
import numpy as np

class Model(object):
	"""RNN Model with ICA preprocessing"""
	def __init__(self, arg, trainable=False):
		self.arg = arg

		self.input_data = tf.placeholder(tf.float32, [arg.batch_size, arg.input_length])
		# target length is 1
		self.label_data = tf.placeholder(tf.float32, [arg.batch_size, 1])

		if arg.lstm:
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(arg.num_units, state_is_tuple=True)
		else:
			self.cell = tf.nn.rnn_cell.GRUCell(arg.num_units)

		if arg.num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * arg.num_layers, state_is_tuple=True)
		self.cell_state = self.cell.zero_state(arg.batch_size, tf.float32)
		self.step_count = tf.Variable(0, trainable=False)

		# RNN cell update
		outputs, self.cell_state = self.cell(self.input_data, self.cell_state)

		# Map the result to a single scalar
		self.mappingW = tf.Variable(tf.random_uniform([arg.num_units, 1], minval=-1, maxval=1, dtype=tf.float32))
		self.pressure = tf.tanh(tf.matmul(outputs, self.mappingW))

		if trainable:
			self.loss = tf.squared_difference(self.pressure, self.label_data)
			trainable_vars = tf.trainable_variables()
			clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), arg.gradient_clip)

			self.trainer = tf.train.AdagradOptimizer(arg.learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.step_count)

		self.saver = tf.train.Saver()
		
	def step(self, session, input_data, label_data=None, trainable=False):
		if trainable:
			input_feed = {self.input_data: input_data, self.label_data:label_data}
			output_var = [self.loss, self.trainer]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.pressure]
		
		output = session.run(output_var, feed_dict=input_feed)

		print session.run(self.label_data, feed_dict=input_feed)

		return output[0]

	def save(self, session, file):
		save_path = self.saver.save(session, file)
		print "Model saved at {}".format(save_path)

	def load(self, session, file):
		self.saver.restore(session, file)
		print "Model loaded from {}".format(file)