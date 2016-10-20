from __future__ import division
import argparse

parser = argparse.ArgumentParser(description="LSTM/GRU sampler")
parser.add_argument('-D', '--data', type=str)
parser.add_argument('-L', '--load', type=str)
parser.add_argument('-S', '--save', type=str)
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
parser.add_argument('-x', dest='execute', action='store_true', help='Execute trades')
parser.set_defaults(verbose=False, execute=False)
arg = parser.parse_args()

import os
import sys
if not os.path.isdir('save/'+arg.load):
	print 'Load file {} does not exist'.format('save/'+arg.load)
	sys.exit(-1)

import pickle
arg.load = 'save/{}/{}'.format(arg.load, arg.load)
saved_arg = pickle.load(open(arg.load+'.cfg', 'rb'))
saved_arg.data = arg.data
saved_arg.load = arg.load
saved_arg.verbose = arg.verbose
saved_arg.save = arg.save
saved_arg.batch_size = 1

if saved_arg.verbose:
	print 'Loading dependencies...'
import lstm
import data_processor
import tensorflow as tf
import numpy as np
import execute
if saved_arg.verbose:
	print 'Finish loading dependencies'

if saved_arg.verbose:
	print 'Loading data...'
_,price,_,_ = data_processor.parse(saved_arg.data)
batch_input,_ = data_processor.get_batch_data(saved_arg, price)
if saved_arg.verbose:
	print 'Finish loading data...'

model = lstm.Model(saved_arg, trainable=False)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	model.load(sess, saved_arg.load+'.model')
	predictions = []
	for b in batch_input:
		pred = np.argmax(model.step(sess,b), axis=1)
		predictions.append(int(pred[0]))
	data_processor.write_label(saved_arg.save, saved_arg.data, predictions, saved_arg.input_length)
if arg.execute:
	print 'Executing the sample...'
	execute.strict_execute_points(saved_arg.save)