import websocket as ws

def on_open(ws):
    ws.send("open")

def on_close(ws):
	print("closed")

def on_message(ws, msg):
	print("msg")

def on_error(ws, er):
	print(er)

ws.enableTrace(True)
socket = ws.WebSocketApp("192.168.1.103", on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
socket.run_forever(http_proxy_port=1109)