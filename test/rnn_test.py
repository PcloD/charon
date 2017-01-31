import sys
sys.path.append('..')
from bunch import Bunch
import numpy as np
import rnn
import tensorflow as tf
import matplotlib.pyplot as plt

arg = Bunch()
arg.batch_size = 1
arg.input_length = 1
arg.lstm = False
arg.num_layers = 3
arg.num_units = 600
arg.learning_rate = 0.001
arg.gradient_clip = 10

model = rnn.Model(arg, trainable=True)

x_test = np.random.normal(scale=0.2,size=100)
y_test = []
for i in range(len(x_test)):
	y_test.append(np.sum(x_test[max(0,i-3):i+1]))
y_test = np.array(y_test).reshape(100)

training_error = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for it in range(1000):
		x = np.random.normal(scale=0.2,size=1000)
		y = []
		for i in range(len(x)):
			y.append(np.sum(x[max(0,i-3):i+1]))

		total_loss = []
		prev_state = model.zero_state()
		for i in range(1000):
			_,loss,prediction,state = model.step(sess, np.array([[x[i]]]), np.array([y[i]]), True, state=prev_state)
			total_loss.append(loss)
			prev_state = state
		print ("Iteration {} Loss {}".format(it, np.mean(total_loss)))
		training_error.append(np.mean(total_loss))

	pred = []
	prev_state = model.zero_state()
	for i in range(100):
		p,state = model.step(sess,np.array([[x_test[i]]]),state=prev_state)
		pred.append(p)
		prev_state = state

	pred = np.array(np.concatenate(pred))
	s = pred - y_test
	x = np.square(s)
	print (np.mean(x))

training_error_x = [i for i in range(len(training_error))]
plt.plot(training_error_x,training_error)
plt.show()