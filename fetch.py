from OkcoinSpotAPI import OKCoinSpot
import datetime
import time

apikey = ''#'7c6f007b-15c9-462f-995f-9e2f14951315'
secretkey = ''#'0BC2EAEE314FFFCFC1A2E26869CE7714'
okcoinRESTURL = 'www.okcoin.com'

okcoinSpot = OKCoinSpot(okcoinRESTURL,apikey,secretkey)

d = okcoinSpot.kline(symbol='btc_usd',time='1hour',since='1264767668',size=3000)