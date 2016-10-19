#############################################################################
#PARSING PRICE FILES
#raw columns are:
# Date | Time | Open | High | Low | Close | Volume (BTC) | Volume(Currency) | Weighted Price

import numpy as np

def parse(data_location):
    raw=np.loadtxt(data_location,dtype=str)
    dlength=raw.shape[0]            #total length of table
    data=np.zeros([dlength,raw.shape[1]-2])
    for row in range(raw.shape[0]):
        for col in range(2,raw.shape[1]):
            data[row][col-2]=float(raw[row][col])
    #weighted price
    price_w=[]
    for i in range(dlength):
        price_w.append(data[i][6])
    #open price
    price_o=[]
    for i in range(dlength):
        price_o.append(data[i][0])
    #close price
    price_c=[]
    for i in range(dlength):
        price_c.append(data[i][3])
    #volume
    vol=[]
    for i in range(dlength):
        vol.append(data[i][4])
    return price_o, price_c, price_w, vol


##############################################################################
#PARSING SWAP FILES
#raw columns are:
# Date | Time | Rate | Total Swap

def parse_swap(data_location):
    raw=np.loadtxt(data_location,dtype=str)
    dlength=raw.shape[0] 
    data=np.zeros([dlength,raw.shape[1]-2])
    for row in range(raw.shape[0]):
         for col in range(2,raw.shape[1]):
            data[row][col-2]=float(raw[row][col])
    #swap rate
    swap_rate=[]
    for i in range(dlength):
        swap_rate.append(data[i][0])
    #swap total
    swap_total=[]
    for i in range(dlength):
        swap_total.append(data[i][1])
    return [swap_rate[::-1], swap_total[::-1]]  #reversing list since bfxdata gives out reversed data

def write_label(label_file, price_file, labels):
    f = open(label_file, 'w')
    data = [price_file, ','.join(map(str,labels))]
    f.writelines(data)

def read_label(label_file):
    f = open(label_file, 'r')
    data = f.readlines()
    _,price,_,_ = parse(data[0].strip('\n'))
    return price, map(int, data[1].split(','))

def get_data(arg, price):
    x,y = [],[]
    for i in xrange(arg.input_length, len(price) - 1):
        x.append(price[i-arg.input_length:i])
        y.append(2 if price[i+1] > (price[i] + arg.price_epsilon) else (1 if price[i+1] < (price[i] - arg.price_epsilon) else 0))
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    x = x - np.expand_dims(np.average(x, axis=1), axis=1)
    overflow = x.shape[0] % arg.batch_size
    x = np.delete(x, range(x.shape[0]-1-overflow, x.shape[0]-1), axis=0)
    y = np.delete(y, range(y.shape[0]-1-overflow, y.shape[0]-1), axis=0)
    num_bin = len(x) / arg.batch_size
    return np.split(x, num_bin), np.split(y, num_bin)