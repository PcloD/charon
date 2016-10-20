import sys
sys.path.append('..')
import data_processor

def get_label_test():
	price = [1,2,3,2,1,5]
	label = [2,0,1,0,2]
	l = data_processor.get_label(price)
	if label == l:
		print '[PASS] data_processor | get_label_test'
	else:
		print '[FAIL] data_processor | get_label_test'
get_label_test()