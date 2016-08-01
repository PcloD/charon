import rnn
import tensorflow as tf
import numpy as np
import parse

batch_size = 50
model = rnn.Model(1200, batch_size=batch_size)

data, _, _, _ = parse.parse('data/data/test2/bfx_hourly.txt')

input_data = data[:-1]
label_data = data[1:]

x = []
y = []
for i in xrange(0, len(input_data), batch_size):
	x.append(np.round(np.array(input_data[i:i+batch_size])))
	y.append(np.round(np.array(label_data[i:i+batch_size])))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for xx,yy in x,y:
	model.step(sess, xx, yy, True)
	model.save(sess, 'saved_model')