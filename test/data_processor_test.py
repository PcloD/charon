import sys
sys.path.append('..')
import data_processor

def get_label_simple_test():
	price = [1,2,3,2,1,5]
	label = [2,0,1,0,2]
	l = data_processor.get_label_simple(price)
	if label == l:
		print '[PASS] data_processor | get_label_simple_test'
	else:
		print '[FAIL] data_processor | get_label_simple_test'

def local_extrema_test():
	price=[1,2,3,4,3,2,1,2,3]
	ex = data_processor.local_extrema(price)
	if ex == [0,3,6,8]:
		print '[PASS] data_processor | local_extrema_test'
	else:
		print '[FAIL] data_processor | local_extrema_test'

def get_label_test():
	price=[1,2,3,4,3,4,2,1,1,4]
	label = [2,0,0,0,0,1,0,0,2,1]
	l = data_processor.get_label(price, 2)
	if label == l:
		print '[PASS] data_processor | get_label_test'
	else:
		print '[FAIL] data_processor | get_label_test'

if __name__ == '__main__':
	get_label_simple_test()
	local_extrema_test()
	get_label_test()