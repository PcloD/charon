from __future__ import division

import argparse

def get_data(arg, price):
	x,y = [],[]
	for i in xrange(arg.input_length, len(price) - 1):
		x.append(price[i-arg.input_length:i])
		y.append(2 if price[i+1] > (price[i] + arg.price_epsilon) else (1 if price[i+1] < (price[i] - arg.price_epsilon) else 0))
	assert len(x) == len(y)
	x = np.array(x)
	y = np.array(y)
	x = x - np.expand_dims(np.average(x, axis=1), axis=1)
	overflow = x.shape[0] % arg.batch_size
	x = np.delete(x, range(x.shape[0]-1-overflow, x.shape[0]-1), axis=0)
	y = np.delete(y, range(y.shape[0]-1-overflow, y.shape[0]-1), axis=0)
	num_bin = len(x) / arg.batch_size
	return np.split(x, num_bin), np.split(y, num_bin)

def train(arg):
	model = lstm.Model(arg, trainable=True)

	if arg.verbose:
		print 'Loading training data...'
	_, price, _, _ = data_parser.parse(arg.data)
	batch_input, batch_output = get_data(arg, price)

	if arg.verbose:
		print 'Finish loading data'

	if arg.verbose:
		label = [2 if price[k+1] > (price[k] + arg.price_epsilon) else (1 if (price[k+1] < price[k] - arg.price_epsilon) else 0) for k in xrange(len(price) - 1)]
		label = np.array(label)
		buy = np.where(label == 2)[0].size
		sell = np.where(label == 1)[0].size
		hold = np.where(label == 0)[0].size
		print 'Label data distribution'
		print "Buy: {} ({:2.4f}%)".format(buy, buy/label.size * 100)
		print "Sell: {} ({:2.4f}%)".format(sell, sell/label.size * 100)
		print "Hold: {} ({:2.4f}%)".format(hold, hold/label.size * 100)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		if arg.load is not None:
			model.load(sess, arg.load)

		for it in xrange(arg.iter):
			total_loss = 0.0
			for i in xrange(len(batch_input) - 1):
				loss = model.step(sess, batch_input[i], batch_output[i], trainable=True)
				total_loss += np.sum(loss) / arg.batch_size

			accuracy = []
			buy=sell=hold=0
			for i in xrange(len(batch_input)):
				pred = model.step(sess, batch_input[i])
				pred = np.argmax(pred, axis=1)
				acc = accuracy_score(batch_output[i], pred)
				accuracy.append(acc)

				buy += np.where(pred == 2)[0].size
				sell += np.where(pred == 1)[0].size
				hold += np.where(pred == 0)[0].size
			print "Iteration {} with average loss {} and accuracy {}".format(it, total_loss / len(batch_input), np.sum(accuracy) / len(accuracy))
			print 'Buy:{} Sell:{} Hold:{}'.format(buy,sell,hold)
			if arg.save is not None:
				model.save(sess, 'save/model/'+arg.save)

#=====================================================================================================================

parser = argparse.ArgumentParser(description="Multilayer RNN trainer")
parser.add_argument('-D', '--data', type=str, help='Historical price data file to be parsed', required=True)
parser.add_argument('-L', '--load', type=str, default=None, help='Load existing model')
parser.add_argument('-S', '--save', type=str, default=None, help='Model save name')
parser.add_argument('--iter', type=int, default=10, help='Maximum number of iterations')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_units', type=int, default=400)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--input_length', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--gradient_clip', type=float, default=5.0)
parser.add_argument('-l', dest='lstm', action='store_true', help='Use LSTM')
parser.add_argument('-g', dest='lstm', action='store_false', help='Use GRU (default)')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
parser.set_defaults(lstm=False, verbose=False)

parser.add_argument('--price_epsilon', type=float, default=0.5)

arg = parser.parse_args()

if arg.verbose:
	print 'Importing libraries...'

import data_parser
import lstm
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import sys

if arg.verbose:
	print 'Finish importing'

if arg.save is not None:
	pickle.dump(arg, open('save/model/'+arg.save+'.cfg', 'wb'))

if arg.load is not None:
	if arg.verbose:
		print 'Loading config from save/model/'+arg.load+'.cfg'
	a = pickle.load(open('save/model/'+arg.load+'.cfg', 'rb'))
	arg.num_units = a.num_units
	arg.num_layers = a.num_layers
	arg.input_length = a.input_length
	arg.lstm = a.lstm
	arg.price_epsilon = a.price_epsilon

train(arg)