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
arg.num_units = 500
arg.learning_rate = 0.000001
arg.gradient_clip = 5

model = rnn.Model(arg, trainable=True)

x_test = np.random.normal(scale=0.2,size=100)
y_test = []
for i in range(len(x_test)):
	y_test.append(np.sum(x_test[max(0,i-3):i+1]))

y_test = np.array(y_test).reshape(100)
training_error = []
test_error = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for it in range(500):
		try:
			x = np.random.normal(scale=0.2,size=10000)
			y = []
			for i in range(len(x)):
				y.append(np.sum(x[max(0,i-3):i+1]))

			total_loss = []
			prev_state = model.zero_state()
			for i in range(1000):
				_,loss,prediction,state = model.step(sess, x[i].reshape(arg.batch_size, arg.input_length), y[i].reshape(arg.batch_size), True, state=prev_state)
				total_loss.append(loss)
				prev_state = state

			print ("Iteration {} Loss {}".format(it, np.mean(total_loss)))
			training_error.append(np.mean(total_loss))

			pred = []
			prev_state = model.zero_state()
			for i in range(len(x_test)):
				p,state = model.step(sess,x_test[i].reshape(arg.batch_size, arg.input_length),state=prev_state)
				pred.append(p)
				prev_state = state

			pred = np.array(np.concatenate(pred)).reshape(100)
			s = pred - y_test
			x = np.square(s)
			test_error.append(np.mean(x))
		except KeyboardInterrupt:
			break

training_error_x = [i for i in range(len(training_error))]
plt.plot(training_error_x,training_error,c='red')
plt.plot(training_error_x,test_error,c='green')
plt.show()

print (pred[:10])
print (y_test[:10])