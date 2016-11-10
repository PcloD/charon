from __future__ import division
import argparse
import sys
import os

def train(arg):
	model = lstm.Model(arg, trainable=True)

	if arg.verbose:
		print 'Loading training data...'
	price = data_processor.parse_file(arg.data)
	batch_input, batch_output = data_processor.get_batch_data(arg, price)

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
		try:
			sess.run(tf.initialize_all_variables())

			if arg.load is not None:
				model.load(sess, arg.load+'.model')

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
				print "Iteration {} with average loss {} and accuracy {}".format(sess.run(model.step_count), total_loss / len(batch_input), np.sum(accuracy) / len(accuracy))
				if arg.verbose:
					print 'Buy:{} Sell:{} Hold:{}'.format(buy,sell,hold)
				if arg.save is not None:
					if arg.save_freq != 0 and it % arg.save_freq == arg.save_freq - 1:
						model.save(sess, arg.save+'.model')
			if arg.save is not None:
				model.save(sess, arg.save+'.model')
		except KeyboardInterrupt:
			print 'Training interrupted by user'
			if arg.save is not None:
					model.save(sess, arg.save+'.model')

#=====================================================================================================================

parser = argparse.ArgumentParser(description="Multilayer RNN trainer")
parser.add_argument('-D', '--data', type=str, help='Historical price data file to be parsed', required=True)
parser.add_argument('-L', '--load', type=str, default=None, help='Load existing model')
parser.add_argument('-S', '--save', type=str, default=None, help='Model save package name')
parser.add_argument('--iter', type=int, default=10, help='Maximum number of iterations')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_units', type=int, default=400)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--input_length', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--gradient_clip', type=float, default=5.0)
parser.add_argument('--save_freq', type=int, default=0)
parser.add_argument('-l', dest='lstm', action='store_true', help='Use LSTM')
parser.add_argument('-g', dest='lstm', action='store_false', help='Use GRU (default)')
parser.add_argument('-f', dest='force', action='store_true', help='Force overwrite existing save files')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
parser.set_defaults(lstm=False, verbose=False, force=False)

parser.add_argument('--price_epsilon', type=float, default=0.5)

arg = parser.parse_args()

if arg.load is None and arg.save is not None:
	if os.path.isdir('save/'+arg.save) and not arg.force:
		print 'Save file already exists, to overwrite, use -f flag'
		sys.exit(0)

if arg.verbose:
	print 'Loading dependencies...'

import data_processor
import lstm
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score


if arg.verbose:
	print 'Finish loading dependencies'

if arg.save is not None:
	if not os.path.isdir('save/'+arg.save):
		os.mkdir('save/'+arg.save)
	arg.save = 'save/{}/{}'.format(arg.save, arg.save)
	pickle.dump(arg, open(arg.save+'.cfg', 'wb'))
	if arg.verbose:
		print 'Configuration saved at ' + arg.save

if arg.load is not None:
	arg.load = 'save/{}/{}'.format(arg.load, arg.load)
	a = pickle.load(open(arg.load+'.cfg', 'rb'))
	arg.num_units = a.num_units
	arg.num_layers = a.num_layers
	arg.input_length = a.input_length
	arg.lstm = a.lstm
	arg.price_epsilon = a.price_epsilon
	if arg.verbose:
			print 'Config loaded from ' + arg.load + '.cfg'
train(arg)