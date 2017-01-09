import execute

file = 'data/btce_hourly_201607_201609.csv'

statfile = open('save/stats.csv', 'w')
initial = 10000
for i in range(1000):
	equity,capital,btc,maxdrawdown,num_trades = execute.random_execute_points(file, initial=initial, trans=None, draw=False)
	s = '{},{}\n'.format(equity-initial, (equity-initial)/initial*100)
	statfile.write(s)
statfile.close