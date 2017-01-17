import sys
sys.path.append('..')
import data_processor
from bunch import Bunch

def get_batch_data_test():
	arg = Bunch()
	arg.price_epsilon = 1
	arg.input_length = 4
	arg.batch_size = 3
	price = [1,2,3,4,3,2,1,4,5]
	x,y = data_processor.get_batch_data(arg, price)
	assert len(x) == len(y)

def get_label_simple_test():
	price = [1,2,3,2,1,5]
	label = [2,0,1,0,2]
	l = data_processor.get_label_simple(price)
	if label == l:
		print ('[PASS] data_processor | get_label_simple_test')
	else:
		print ('[FAIL] data_processor | get_label_simple_test')

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

def parse_test():
	arg = Bunch()
	arg.price_epsilon = 1
	arg.input_length = 4
	arg.batch_size = 3
	label = [[881.7,881.7,881.7,881.7,0.0],[881.27,881.45,881.27,881.44,1.0],[881.66,885.0,881.66,885.0,38.732]]
	loaded_label = data_processor.parse('../data/okc_data_1m.csv')
	batch_input, batch_output = data_processor.get_batch_data(arg, loaded_label)
	print (batch_input[0])
	if label == loaded_label[:3]:
		print ('[PASS] data_processor | parse_test')
	else:
		print ('[FAIL] data_processor | parse_test')

if __name__ == '__main__':
	get_label_simple_test()
	local_extrema_test()
	get_label_pressure()
	get_label_test()
	get_batch_data_test()
	parse_test()