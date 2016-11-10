import datetime
import sys
import os

arg = sys.argv
if len(arg) < 3:
	print 'Missing file name'
	sys.exit(0)

last_time = 0
outputfilename = arg[2] + '_hourly.csv'
with open(arg[1], 'r') as infile:
	with open(outputfilename, 'w') as outfile:
		for line in infile:
			tok = line.split(',')
			tok[0] = int(tok[0])
			if tok[0] - last_time > 3600:
				last_time = tok[0]
				tok[0] = str(datetime.datetime.fromtimestamp(tok[0]))
				outfile.write(','.join(tok[:-1]) + '\n')

