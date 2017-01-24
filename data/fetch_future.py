from HttpMD5Util import httpGet
import datetime
import time
import os

rest_url = 'www.okcoin.com'
ticker_uri = '/api/v1/future_ticker.do'
index_price_uri = '/api/v1/future_index.do'
hold_amount_uri = '/api/v1/future_hold_amount.do'
depth_uri = '/api/v1/future_depth.do'

data_file_name = 'okc_data_1m.csv'

def fetch_depth():
	params = 'symbol=btc_usd&contract_type=quarter&size=50'
	return httpGet(rest_url,depth_uri,params)

def fetch_index_price():
	params = 'symbol=btc_usd&contract_type=quarter'
	return httpGet(rest_url,index_price_uri,params)

def fetch_future_hold_amount():
	params = 'symbol=btc_usd&contract_type=quarter'
	return httpGet(rest_url,hold_amount_uri,params)

def fetch_ticker():
	params = 'symbol=btc_usd&contract_type=quarter'
	return httpGet(rest_url,ticker_uri,params)

def parse_depth(depth, last):
	ask1,ask2,ask3,bid1,bid2,bid3 = 0,0,0,0,0,0
	depth['asks'].reverse()
	for ask in depth['asks']:
		if ask[0] - last < 1:
			ask1 += ask[1]
		if ask[0] - last < 2:
			ask2 += ask[1]
		if ask[0] - last < 3:
			ask3 += ask[1]
	for bid in depth['bids']:
		if bid[0] - last > -1:
			bid1 += bid[1]
		if bid[0] - last > -2:
			bid2 += bid[1]
		if bid[0] - last > -3:
			bid3 += bid[1]
	return ask1,ask2,ask3,bid1,bid2,bid3

file = open('okc_future_data_15s.csv', 'a')


try:
	while True:
		timestamp = int(time.time() * 1000)
		if timestamp % 15000 != 0:	# every 10 seconds
			continue

		ticker = fetch_ticker()['ticker']
		hold = fetch_future_hold_amount()
		index = fetch_index_price()
		depth = fetch_depth()

		ask1,ask2,ask3,bid1,bid2,bid3 = parse_depth(depth,ticker['last'])

		expiration = str(ticker['contract_id'])[2:8]
		expiration_d = datetime.datetime.strptime(expiration, '%y%m%d')
		expiration_t = int(time.mktime(expiration_d.timetuple()) * 1000)

		line = [timestamp,expiration_t,ticker['last'],ticker['high'],ticker['low'],ticker['buy'],ticker['sell'],hold[0]['amount'],index['future_index'],ask1,ask2,ask3,bid1,bid2,bid3]
		line = [str(i) for i in line]
		file.write(','.join(line) + '\n')
		file.flush()
		os.fsync(file)
except KeyboardInterrupt:
	file.flush()
	os.fsync(file)