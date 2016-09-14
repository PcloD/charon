import tensorflow as tf
import numpy as np
import argparse

import data_parser
import lstm

def main():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--training_data', type=str, default='Data/Data/mtgox_hourly.txt', help='Historical price data file to be parsed')
	parser.add_argument('--save_dir', type=str, default='save', help='Model save directory')
	parser.add_argument('--lstm', type=bool, default=False)
	parser.add_argument('--batch_size', type=int, default=50)
	parser.add_argument('--num_units', type=int, default=256)
	parser.add_argument('--num_layers', type=int, default=2)
	parser.add_argument('--input_length', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.1)
	parser.add_argument('--gradient_clip', type=float, default=5.0)

	parser.add_argument('--price_epsilon', type=float, default=1.0)

	arg = parser.parse_args()
	train(arg)

def binning(l, k):
	for i in xrange(0, len(l), k):
		yield np.array([l[i:i+k]]).T

def get_input_data(arg):
	_, price, _, _ = data_parser.parse(arg.training_data)
	d = list(binning(price, arg.batch_size))
	if len(d[-1]) < arg.batch_size:
		return d[:-1]
	else:
		return d

def get_output_data(arg):
	_, price, _, _ = data_parser.parse(arg.training_data)
	label = [1.0 if price[k+1] > (price[k] + arg.price_epsilon) else (-1.0 if (price[k+1] < price[k] - arg.price_epsilon) else 0.0) for k in xrange(len(price) - 1)]
	d = list(binning(label, arg.batch_size))
	if len(d[-1]) < arg.batch_size:
		return d[:-1]
	else:
		return d

def train(arg):
	model = lstm.Model(arg, trainable=True)
	batch_input = get_input_data(arg)
	batch_output = get_output_data(arg)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for i in xrange(len(batch_input)):
			loss = model.step(sess, batch_input[i], batch_output[i], trainable=True)
			print "Iteration {} with loss {}".format(i, np.sum(loss))

if __name__ == '__main__':
	main()