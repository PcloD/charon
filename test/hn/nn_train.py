from __future__ import division

import numpy as np
import tensorflow as tf
import nn
from sklearn.metrics import accuracy_score

def binrep(n,r):
    return "{0:0{1}b}".format(n, r)

x = np.random.randint(0,10,size=(1000,2))
y = np.sum(x, axis=1)
xtemp = []
for i in xrange(x.shape[0]):
	xtemp.append(list(binrep(x[i][0], 4)) + list(binrep(x[i][1],4)))
x = np.array(xtemp)
xt = np.random.randint(0,10,size=(200,2))
yt = np.sum(xt,axis=1)
xtemp = []
for i in xrange(xt.shape[0]):
	xtemp.append(list(binrep(xt[i][0], 4)) + list(binrep(xt[i][1],4)))
xt = np.array(xtemp)

with tf.Session() as sess:
	a = []
	for i in xrange(10):
		model = nn.Model(8, 19, [64], True, 0.1)
		sess.run(tf.initialize_all_variables())
		for i in xrange(1000):
			loss = model.step(sess, x, y, trainable=True)

		pred = model.step(sess, xt)
		pred = np.argmax(pred, axis=1)
		acc = accuracy_score(pred, yt)
		print 'Accuracy: {}'.format(acc)
		a.append(acc)
	print 'Avg Accuracy: {}'.format(np.mean(a))