import tensorflow as tf
import numpy as np

class Model:
	def __init__ (self, trainable=True, input_length=1, batch_size=1, num_units=256, num_layers=1, use_lstm=False, learning_rate=0.1, gradient_clip=5.0):
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

		inputs = tf.split(split_dim=1, num_split=input_length, value=self.input_data)
		inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

		outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, self.cell)

		self.output = outputs[0]

		if trainable:
			self.loss = tf.square(self.label_data - self.output)

			trainable_vars = tf.trainable_variables()
			clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), gradient_clip)

			self.trainer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.step_count)

	def step(self, session, input_data, label_data=None, training=False):
		if training:
			input_feed = {self.input_data: input_data, self.label_data:label_data}
			output_var = [self.trainer, self.loss]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.output]

		final_output = session.run(output_var, feed_dict=input_feed)
		
		if training:
			return final_output[1]
		else:
			return final_output[0]

