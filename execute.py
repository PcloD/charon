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
	print 'Final capital: {} BTC: {}'.format(capital, btc)
	print 'Final equity: {}'.format(capital + btc * price[-1])

	plt.plot(price)
	bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
	plt.scatter(bx, by, c='green', s=50)
	plt.scatter(sx, sy, c='red', s=50)
	plt.show()

def strict_execute_points(file, input_length, initial=1000, hold=0, sell=1, buy=2):
	price, actions = data_processor.read_label(file)
	assert len(price) == len(actions) + input_length + 1
	capital = initial
	btc = 0
	for i in xrange(len(actions)):
		if actions[i] == 1 and btc > 0:
			capital += price[i+input_length]
			btc -= 1
		elif actions[i] == 2 and capital - price[i] > 0:
			capital -= price[i+input_length]
			btc += 1
	print 'Final capital: {} BTC: {}'.format(capital, btc)
	print 'Final equity: {}'.format(capital + btc * price[-1])

	plt.plot(price)
	bx, by, sx, sy = get_action_points(price, input_length, actions, sell, buy)
	plt.scatter(bx, by, c='green', s=50)
	plt.scatter(sx, sy, c='red', s=50)
	plt.show()

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

execute_points('save/output', 100)