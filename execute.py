from __future__ import division
import data_processor
import matplotlib.pyplot as plt

def execute_points(file, input_length, hold=0, sell=1, buy=2):
	price, actions = data_processor.read_label(file)
	capital = 0
	btc = 0
	for i in xrange(len(actions)):
		if actions[i] == 1:
			capital += price[i+input_length]
			btc -= 1
		elif actions[i] == 2:
			capital -= price[i+input_length]
			btc += 1
		if capital + btc*price[i] < 0:
			print 'Dragdown at time {}: {}'.format(i, capital + btc * price[i])
	equity = capital + btc * price[-1]
	print 'Final equity: {} ({}$ {}BTC)'.format(equity, capital, btc)

	bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
	draw_buysell(price, bx, by, sx, sy)

def strict_execute_points(file, initial=1000, hold=0, sell=1, buy=2):
	price, input_length, actions = data_processor.read_label(file)
	print len(price), len(actions), input_length
	assert len(price) == len(actions) + input_length + 1
	capital = initial
	btc = 0
	maxdrawdown = 0
	for i in xrange(len(actions)):
		if actions[i] == 1 and btc > 0:
			capital += price[i+input_length]
			btc -= 1
		elif actions[i] == 2 and capital - price[i] > 0:
			capital -= price[i+input_length]
			btc += 1
		if capital + btc * price[i] - initial < maxdrawdown:
			maxdrawdown = capital + btc * price[i] - initial
	equity = capital + btc * price[-1]
	print 'Final Equity: {} ({}$ {}BTC)'.format(equity, capital, btc)
	print 'P/L: {}({}%) | Max Drawdown: {}'.format(equity - initial, (equity-initial)/initial*100,maxdrawdown)

	bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
	draw_buysell(price, bx, by, sx, sy)

def get_action_points(price, input_length, actions, sell=1, buy=2):
	buy_x = []
	buy_y = []
	sell_x = []
	sell_y = []
	for i in xrange(len(actions)):
		if actions[i] == buy:
			buy_x.append(i+input_length)
			buy_y.append(price[i+input_length])
		elif actions[i] == sell:
			sell_x.append(i+input_length)
			sell_y.append(price[i+input_length])
	return buy_x, buy_y, sell_x, sell_y

def draw_buysell(price, buy_x, buy_y, sell_x, sell_y):
	plt.plot(price)
	plt.scatter(buy_x, buy_y, c='green', s=10, marker='x')
	plt.scatter(sell_x, sell_y, c='red', s=10, marker='o')
	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('file', nargs=1)
	parser.add_argument('-i', dest='initial', type=int, help='Initial capital')
	arg = parser.parse_args()
	strict_execute_points(arg.file[0], initial=arg.initial)