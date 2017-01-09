from HttpMD5Util import *

class OKCoinSpot:
    def __init__(self,url,apikey,secretkey):
        self.__url = url
        self.__apikey = apikey
        self.__secretkey = secretkey

    def ticker(self,symbol = ''):
        TICKER_RESOURCE = "/api/v1/ticker.do"
        params=''
        if symbol:
            params = 'symbol=%(symbol)s' %{'symbol':symbol}
        return httpGet(self.__url,TICKER_RESOURCE,params)

    def depth(self,symbol = ''):
        DEPTH_RESOURCE = "/api/v1/depth.do"
        params=''
        if symbol:
            params = 'symbol=%(symbol)s' %{'symbol':symbol}
        return httpGet(self.__url,DEPTH_RESOURCE,params) 

    def trades(self,symbol = ''):
        TRADES_RESOURCE = "/api/v1/trades.do"
        params=''
        if symbol:
            params = 'symbol=%(symbol)s' %{'symbol':symbol}
        return httpGet(self.__url,TRADES_RESOURCE,params)
    
    def kline(self,symbol='btc_usd',time='1min',since='1417536000000',size='5000'):
        TRADES_RESOURCE = '/api/v1/kline.do'
        params=''
        params = 'symbol=%(symbol)s&type=%(type)s&since=%(since)s&size=%(size)s' % {'symbol':symbol, 'type':time,'since':since,'size':size}
        return httpGet(self.__url, TRADES_RESOURCE,params)












    
