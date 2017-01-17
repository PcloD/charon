import tensorflow as tf
import numpy as np
import supercell

class Model(object):
	def __init__(self, arg, trainable=False):
		self.arg = arg
		output_dim = 1

		self.input_data = tf.placeholder(tf.float32, [arg.batch_size, arg.input_length])
		self.label_data = tf.placeholder(tf.float32, [arg.batch_size])

		if arg.lstm:
			self.cell = supercell.HyperLSTMCell(arg.num_units)
		else:
			self.cell = tf.nn.rnn_cell.GRUCell(arg.num_units)

		if arg.num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * arg.num_layers, state_is_tuple=True)
		self.cell_state = self.cell.zero_state(arg.batch_size, tf.float32)

		# RNN cell update
		outputs, self.cell_state = self.cell(self.input_data, self.cell_state)

		# Map the result to a single scalar
		self.softmaxW = tf.Variable(tf.random_uniform([arg.num_units, output_dim], minval=-1, maxval=1, dtype=tf.float32))
		self.softmaxb = tf.Variable(tf.truncated_normal([1, output_dim]), dtype=tf.float32)
		self.prediction = tf.tanh(tf.matmul(outputs, self.softmaxW) + self.softmaxb)

		if trainable:
			self.loss = tf.squared_difference(self.prediction, self.label_data)
			trainable_vars = tf.trainable_variables()
			clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), arg.gradient_clip)

			self.trainer = tf.train.AdamOptimizer(arg.learning_rate).apply_gradients(zip(clipped_grads, trainable_vars))

		self.saver = tf.train.Saver()
		
	def step(self, session, input_data, label_data=None, trainable=False):
		if trainable:
			input_feed = {self.input_data: input_data, self.label_data:label_data}
			output_var = [self.trainer, self.loss, self.prediction]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.prediction]
		
		output = session.run(output_var, feed_dict=input_feed)
		return output[1] if trainable else output[0]

	def save(self, session, file):
		save_path = self.saver.save(session, file)
		print("Model saved at {}".format(save_path))

	def load(self, session, file):
		self.saver.restore(session, file)
		print("Model loaded from {}".format(file))