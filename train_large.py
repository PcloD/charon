import rnn
import numpy as np
import parse

import tensorflow as tf

batch_size = 50
model = rnn.Model(1200, batch_size=batch_size)

start_line = 0
end_line = 51
line_counter = -1
training_session = 0
data_batch = []

sess = tf.Session()
sess.run(tf.initialize_all_variables())
with open("Data/Data/btceUSD.csv") as file:
	for line in file:
		line_counter += 1
		if line_counter < start_line:
			continue
		if line_counter >= end_line:
			break
		
		tokens = line.split(',')
		data_batch.append(float(tokens[1]))

		if (len(data_batch) > batch_size):
			x = np.round(np.array(data_batch[:-1])).reshape([batch_size, 1])
			y = np.round(np.array(data_batch[1:])).reshape([batch_size, 1])

			data_batch = []

			loss = model.step(sess, x, y, True)

			training_session += 1
			if training_session % 100 == 0:
				model.save(sess, 'save_model')
				print 'Iteration {0} with loss {1}'.format(training_session, loss)

x = np.array([636]).reshape([batch_size, 1])
print model.step(sess, x)