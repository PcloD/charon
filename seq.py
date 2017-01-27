import tensorflow as tf
import numpy as np

class Model(object):
	def __init__(self, arg, trainable=False):
		self.arg = arg
		output_dim = 1

		self.input_data = tf.placeholder(tf.float32, [arg.batch_size, arg.input_length])
		self.label_data = tf.placeholder(tf.float32, [arg.batch_size, output_dim])

		if arg.lstm:
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(arg.num_units)
		else:
			self.cell = tf.nn.rnn_cell.GRUCell(arg.num_units)

		if arg.num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * arg.num_layers, state_is_tuple=True)
		self.cell_state = self.cell.zero_state(arg.batch_size, tf.float32)

		self.output,self.cell_state = tf.nn.seq2seq.rnn_decoder(self.input_data, self.cell_state, self.cell)
		