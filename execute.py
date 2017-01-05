from __future__ import division
import data_processor
import numpy as np
import matplotlib.pyplot as plt

def execute_points(file, hold, sell, buy):
	""" Execute every point in the label file regardless of available funds
		Buy/Sell 1 BTC at a time
	"""
	price, input_length, actions = data_processor.read_label(file)
	capital = 0
	btc = 0
	for i in range(len(actions)):
		if actions[i] == 1:
			capital += price[i+input_length]
			btc -= 1
		elif actions[i] == 2:
			capital -= price[i+input_length]
			btc += 1
	equity = capital + btc * price[-1]
	print('Final Equity: {} ({}$ {}BTC)'.format(equity, capital, btc))

	bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
	draw_buysell(price, bx, by, sx, sy)

def strict_execute_points(file, sell, buy, initial=1000, trans=1, draw=True):
	""" Execute every point in the file, does not overdraw
		Buy/Sell 1 BTC at a time
	"""
	if trans is None or trans <= 0:
		trans = 100000000

	price, input_length, actions = data_processor.read_label(file)
	num_trades = 0
	capital = initial
	btc = 0
	maxdrawdown = 0
	for i in range(len(actions)):
		if actions[i] == sell and btc > 0:
			capital += price[i+input_length] * min(trans, btc)
			btc -= min(trans, btc)
			num_trades += 1
		elif actions[i] == buy and capital - price[i] > 0:
			transaction_amount = min(capital / price[i+input_length], trans)
			capital -= price[i+input_length] * transaction_amount
			btc += 1 * transaction_amount
			num_trades += 1
		if capital + btc * price[i] - initial < maxdrawdown:
			maxdrawdown = capital + btc * price[i] - initial
	equity = capital + btc * price[-1]

	print('Final Equity: {} ({}$ {}BTC)'.format(equity, capital, btc))
	print('P/L: {}({}%) | Max Drawdown: {} | Number of trades: {}'.format(equity - initial, (equity-initial)/initial*100,maxdrawdown, num_trades))

	if draw:
		bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
		draw_buysell(price, bx, by, sx, sy)
	return equity, capital, btc, maxdrawdown, num_trades

def random_execute_points(file, initial=1000, trans=1, draw=True):
	price = data_processor.parse_file(file)
	actions = np.random.choice([-1,0,1], size=len(price)-1)
	data_processor.write_label("save/temp_random.csv", file, actions, 1)
	return strict_execute_points("save/temp_random.csv", -1, 1, initial=initial, trans=trans, draw=draw)

def get_action_points(price, input_length, actions, sell=1, buy=2):
	buy_x = []
	buy_y = []
	sell_x = []
	sell_y = []
	for i in range(len(actions)):
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
	parser.add_argument('-i', default=1000, dest='initial', type=int, help='Initial capital')
	parser.add_argument('-m', default='random', type=str, dest='method')
	parser.add_argument('--no_draw', dest='draw', action='store_false', help='Draw')
	parser.set_defaults(draw=False)
	arg = parser.parse_args()
	# execute_points(arg.file[0])
	# strict_execute_points(arg.file[0], initial=arg.initial)
	if arg.method == 'random':
		random_execute_points(arg.file[0], initial=arg.initial, trans=10000000, draw=arg.draw)