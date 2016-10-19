def execute_points(file, hold=0, sell=1, buy=2):
	points = map(int, open(file).read().split(','))
	points
