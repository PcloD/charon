import sys
sys.path.append('..')
from bunch import Bunch
import numpy as np
import rnn
import tensorflow as tf

arg = Bunch()
arg.batch_size = 1
arg.input_length = 1
arg.lstm = False
arg.num_layers = 2
arg.num_units = 200
arg.learning_rate = 0.001
arg.gradient_clip = 2

model = rnn.Model(arg, trainable=True)
x = np.random.normal(scale=0.2,size=1000)
y = []
for i in range(len(x)):
	y.append(np.sum(x[max(0,i-3):i+1]))

x_test = x[:10]
y_test = []
for i in range(len(x_test)):
	y_test.append(np.sum(x_test[max(0,i-3):i+1]))
y_test = np.array(y_test).reshape(10,1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for it in range(100):
		total_loss = []
		for i in range(1000):
			_,loss,prediction = model.step(sess, np.array([[x[i]]]), np.array([[y[i]]]), True)
			total_loss.append(loss)
		print ("Iteration {} Loss {}".format(it, np.mean(total_loss)))

	pred = []
	for i in range(10):
		p = model.step(sess,np.array([[x_test[i]]]))
		pred.append(p)

	pred = np.array(np.concatenate(pred))
	s = pred - y_test
	x = np.square(s)
	print (x)
	print (pred)
	print (y_test)