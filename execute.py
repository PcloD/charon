import data_processor
def execute_points(file, hold=0, sell=1, buy=2):
	price, actions = data_processor.read_label(file)
	print len(price), len(actions)
	capital = 0
	btc = 0
	for p in price:
		if actions == 1:
			capital += p
			btc -= 1
		elif actions == 2:
			capital -= p
			btc += 1
	print 'Final capital: {} btc: {}'.format(capital, btc)
	print 'Final equity: {}'.format(capital + btc * price[-1])

execute_points('save/output')