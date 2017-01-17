from __future__ import division
import argparse
import sys
import os

def train(arg):
	model = lstm.Model(arg, trainable=True)

	if arg.verbose:
		print('Loading training data...')
	price = data_processor.parse_high_frequency(arg.data)
	batch_input, test_input, batch_output, test_output = data_processor.get_batch_data(arg, price)
	if arg.verbose:
		print('Finish loading data')

	with tf.Session() as sess:
		try:
			sess.run(tf.global_variables_initializer())

			if arg.load is not None:
				model.load(sess, arg.load+'.model')

			for it in range(arg.iter):
				# training phase
				total_loss = []
				for i in range(len(batch_input)):
					loss = model.step(sess, batch_input[i], batch_output[i], trainable=True)
					total_loss.append(np.mean(loss))

				if arg.save is not None:
					if arg.save_freq != 0 and it % arg.save_freq == arg.save_freq - 1:
						model.save(sess, arg.save+'.model')
				
				# test phase
				total_test_loss = []
				correct = 0
				print ("testing phase")
				
				for i in range(len(test_input)):
					predict = model.step(sess, test_input[i])
					loss = np.mean((predict - test_output[i]) ** 2)
					total_test_loss.append(loss)
					if predict * test_output[i] > 0:
						correct += 1
				print("Iteration {} | Average training loss {} | Average testing loss {} | Correct guess {}".format(it, np.mean(total_loss), np.mean(total_test_loss), correct))

			if arg.save is not None:
				model.save(sess, arg.save+'.model')
		except KeyboardInterrupt:
			print('Training interrupted by user')
			if arg.save is not None:
					model.save(sess, arg.save+'.model')

#=====================================================================================================================

parser = argparse.ArgumentParser(description="Multilayer RNN trainer")
parser.add_argument('-D', '--data', type=str, help='Historical price data file to be parsed', required=True)
parser.add_argument('-L', '--load', type=str, default=None, help='Load trained package')
parser.add_argument('-S', '--save', type=str, default=None, help='Model save package name')
parser.add_argument('--iter', type=int, default=200, help='Maximum number of iterations')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_units', type=int, default=800)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--input_length', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--gradient_clip', type=float, default=5.0)
parser.add_argument('--save_freq', type=int, default=0)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('-l', dest='lstm', action='store_true', help='Use LSTM')
parser.add_argument('-g', dest='lstm', action='store_false', help='Use GRU (default)')
parser.add_argument('-f', dest='force', action='store_true', help='Force overwrite existing save files')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
parser.set_defaults(lstm=False, verbose=False, force=False)

parser.add_argument('--price_epsilon', type=float, default=0.5)

arg = parser.parse_args()

if arg.load is None and arg.save is not None:
	if os.path.isdir('save/'+arg.save) and not arg.force:
		print('Save file already exists, to overwrite, use -f flag')
		sys.exit(0)

if arg.verbose:
	print('Loading dependencies...')

import data_processor
import lstm
import tensorflow as tf
import numpy as np
import pickle
import execute

if arg.verbose:
	print('Finish loading dependencies')

# expand save path
if arg.save is not None:
	if not os.path.isdir('save/'+arg.save):
		os.mkdir('save/'+arg.save)
	arg.save = 'save/{}/{}'.format(arg.save, arg.save)
	pickle.dump(arg, open(arg.save+'.cfg', 'wb'))
	if arg.verbose:
		print('Configuration saved at ' + arg.save +'.cfg')

# load config
if arg.load is not None:
	arg.load = 'save/{}/{}'.format(arg.load, arg.load)
	a = pickle.load(open(arg.load+'.cfg', 'rb'))
	arg.num_units = a.num_units
	arg.num_layers = a.num_layers
	arg.input_length = a.input_length
	arg.lstm = a.lstm
	arg.price_epsilon = a.price_epsilon
	if arg.verbose:
			print('Config loaded from ' + arg.load + '.cfg')
train(arg)