import websocket
import time
import numpy as np
import csv
import json

connection_url = "wss://real.okcoin.com:10440/websocket/okcoinapi"
kline1m_channel = 'ok_sub_spotusd_btc_kline_1min'
depth_channel = 'ok_sub_spotusd_btc_depth_60'

data_file_name = 'okcoin_data.csv'

data_cache = []

def flush_kline_data(data):
	for d in data:
		timestamp = d[0]

def on_open(ws):
	print ("===== Opening Connection =====")
	request = {}
	request['event'] = 'addChannel'
	request['channel'] = kline1m_channel
	print ('Opening channel to', kline1m_channel)
	ws.send(str(request))
	# time.sleep(1)
	# request['channel'] = depth_channel
	# print ('Opening channel to', depth_channel)
	# ws.send(str(request))

def on_close(ws):
	print ("===== Closing Connection =====")

def on_error(ws, error):
	print ("===== Connection Error =====")
	print (error)

def on_message(ws, message):
	response = json.loads(message)[0]
	if 'success' in response and response['success'] == 'false':
		print ("===== Request Error =====")		
		print (response['channel'], response['errorcode'])
	else:
		if response['channel'] == kline1m_channel:
			data = response['data']
			flush_kline_data(data)
		elif response['channel'] == depth_channel:
			data = response['data']

if __name__ == '__main__':
	ws = websocket.WebSocketApp(connection_url, on_open=on_open, on_close=on_close, on_message=on_message, on_error=on_error)
	ws.run_forever()