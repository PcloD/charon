import tensorflow as tf
import numpy as np

class Model(object):
	def __init__(self, arg, feature_size, trainable=False):
		self.arg = arg
		output_dim = 1

		self.input_data = tf.placeholder(tf.float32, [arg.batch_size, arg.input_length, feature_size])
		self.label_data = tf.placeholder(tf.float32, [arg.batch_size])

		if arg.lstm:
			self.cell = tf.contrib.rnn.BasicLSTMCell(arg.num_units, state_is_tuple=True)
		else:
			self.cell = tf.contrib.rnn.GRUCell(arg.num_units)

		if arg.num_layers > 1:
			self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * arg.num_layers, state_is_tuple=True)

		# RNN cell update
		self.outputs, self.cell_state = tf.nn.dynamic_rnn(self.cell, self.input_data, dtype=tf.float32)

		# Map the result to a single scalar
		self.softmaxW = tf.Variable(tf.random_uniform([arg.num_units, output_dim], minval=-0.005, maxval=0.005, dtype=tf.float32))
		self.softmaxb = tf.Variable(tf.random_uniform([1, output_dim], minval=-0.001, maxval=0.001, dtype=tf.float32))
		self.k = tf.Variable(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.float32))
		self.prediction = self.k * (tf.matmul(self.outputs, self.softmaxW) + self.softmaxb)

		if trainable:
			self.loss = tf.squared_difference(self.label_data, self.prediction)
			trainable_vars = tf.trainable_variables()
			opt = tf.train.AdamOptimizer(arg.learning_rate)
			# clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), arg.gradient_clip)
			# self.trainer = opt.apply_gradients(zip(clipped_grads, trainable_vars))
			self.trainer = opt.minimize(self.loss, var_list=trainable_vars)
		self.saver = tf.train.Saver()

		self.test_label = tf.placeholder(tf.float32, [arg.batch_size])
		self.test_predict = tf.placeholder(tf.float32, [arg.batch_size])
		self.test_loss = tf.nn.l2_loss(self.test_label - self.test_predict)
	
	def zero_state(self):
		if self.arg.lstm:
			return np.zeros((self.arg.num_layers, 2, self.arg.batch_size, self.arg.num_units))
		else:
			return np.zeros((self.arg.num_layers, self.arg.batch_size, self.arg.num_units))

	def step(self, session, input_data, label_data=None, trainable=False):
		input_feed = {}
		input_feed[self.input_data] = input_data
		if trainable:
			input_feed[self.label_data] = label_data
			output_var = [self.trainer, self.loss, self.prediction, self.cell_state]
		else:
			output_var = [self.prediction, self.cell_state]
		output = session.run(output_var, feed_dict=input_feed)
		return output

	def save(self, session, file):
		save_path = self.saver.save(session, file)
		print("Model saved at {}".format(save_path))

	def error(self, session, prediction, label):
		input_feed = {self.test_predict:prediction, self.test_label:label}
		return session.run(self.test_loss, feed_dict=input_feed)

	def load(self, session, file):
		self.saver.restore(session, file)
		print("Model loaded from {}".format(file))