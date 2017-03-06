import sys
sys.path.append('..')
import data_processor
from bunch import Bunch

def get_batch_data_test():
	arg = Bunch()
	arg.batch_size = 1
	arg.test_size = 0
	arg.verbose = False
	price = data_processor.parse_high_frequency('../data/okc_future_data_15s.csv')
	x,xt,y,yt = data_processor.get_batch_data(arg, price)
	assert len(x) == len(y)
	print (list(x[0][0]))

def get_label_pressure():
	price = [1,2,3,4,3, 4, 2, 1,1, 4]
	label = [1,1,1,1,1,-1,-1,-1,1]
	l = data_processor.get_label_pressure(price, 2)
	if label == l:
		print ('[PASS] data_processor | get_label_pressure_test')
	else:
		print ('[FAIL] data_processor | get_label_pressure_test')

def local_extrema_test():
	price=[1,2,3,4,3,2,1,2,3]
	ex = data_processor.local_extrema(price)
	if ex == [0,3,6,8]:
		print ('[PASS] data_processor | local_extrema_test')
	else:
		print ('[FAIL] data_processor | local_extrema_test')

def get_label_test():
	price=[1,2,3,4,3,4,2,1,1,4]
	label = [2,0,0,0,0,1,0,0,2]
	l = data_processor.get_label(price, 2, hold=0, sell=1, buy=2)
	if label == l:
		print ('[PASS] data_processor | get_label_test')
	else:
		print ('[FAIL] data_processor | get_label_test')

if __name__ == '__main__':
	local_extrema_test()
	get_label_pressure()
	get_label_test()
	get_batch_data_test()
