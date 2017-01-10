from HttpMD5Util import httpGet
import datetime
import time
import os

rest_url = 'www.okcoin.com'
kline_uri = '/api/v1/kline.do'

data_file_name = 'okc_data_1m.csv'

def fetch_kline(since):
	params = 'symbol=btc_usd&type=1min&size=3000&since={}'.format(since)
	d = httpGet(rest_url,kline_uri,params)
	return d

if __name__ == '__main__':
	last_time_stamp = '1483938000000'
	datafile = open(data_file_name, 'ab+')
	datafile.seek(-512,1)
	lastline = datafile.readlines()[-1].decode('ascii')
	last_time_stamp = lastline.split(',')[0]
	print (last_time_stamp)
	data = fetch_kline(last_time_stamp)[1:]
	for d in data:
		datafile.write((','.join(str(x) for x in d) + '\n').encode('ascii'))