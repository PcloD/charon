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

def parse_file(data_location):
    file = open(data_location, 'r')
    price_data = []
    for line in file:
        price_data.append(float(line.split(',')[1].rstrip()))
    return price_data

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

def write_label(label_file, price_file, labels, input_length):
    f = open(label_file, 'w')
    data = [price_file+'\n', str(input_length)+'\n', ','.join(map(str,labels))]
    f.writelines(data)

def read_label(label_file):
    f = open(label_file, 'r')
    data = f.readlines()
    price = parse_file(data[0].strip('\n'))
    return price, int(data[1].strip('\n')), map(int, data[2].split(','))

def get_batch_data(arg, price):
    x = []
    for i in xrange(arg.input_length-1, len(price) - 1):
        x.append(price[i-arg.input_length+1:i+1])
    y = get_label(price[arg.input_length-1:], arg.price_epsilon)
    x = np.array(x)
    y = np.array(y)
    x = x - np.expand_dims(np.average(x, axis=1), axis=1)   #centering
    overflow = x.shape[0] % arg.batch_size
    x = np.delete(x, range(x.shape[0]-1-overflow, x.shape[0]-1), axis=0)
    y = np.delete(y, range(y.shape[0]-1-overflow, y.shape[0]-1), axis=0)
    num_bin = len(x) / arg.batch_size
    return np.split(x, num_bin), np.split(y, num_bin)

def get_label_simple(price, hold=0, sell=1, buy=2):
    labels = []
    labels.append(sell if price[0] > price[1] else buy)
    action = labels[-1]
    for i in xrange(1, len(price)-2):
        if price[i+1] > price[i]:
            if action == buy:
                action = hold
            else:
                action = buy
        else:
            if action == sell:
                action = hold
            else:
                action = sell

        labels.append(action)
    labels.append(1 if price[-2] > price[-1] else 2)
    return labels

def get_label(price, price_epsilon, hold=0, sell=1, buy=2):
    ex = local_extrema(price)
    # consider the intervals, remove the intervals that are smaller than price_epsilon
    i = 0
    while True:
        if i >= len(ex) - 1:
            break

        dif = abs(price[ex[i+1]] - price[ex[i]])
        if dif < price_epsilon:
            if i == 0:
                del ex[i+1]
                continue
            if i+1 == len(ex) - 1:
                del ex[i]
                continue
            leftdif = abs(price[ex[i-1]] - price[ex[i]])
            rightdif = abs(price[ex[i+2]] - price[ex[i+1]])
            if leftdif > rightdif:
                del ex[i+1]
            else:
                del ex[i]
        else:
            i += 1
    inc = price[0] < price[1]
    labels = []
    exi = 0
    for i in xrange(len(price)-1):
        if i == ex[exi]:
            labels.append(buy if inc else sell)
            exi += 1
            inc = not inc
        else:
            labels.append(hold)

    return labels

def local_extrema(price):
    """Return the indices of local extrema of the price sequence"""
    local=[]
    local.append(0)
    inc = price[0] < price[1]
    for i in xrange(1, len(price)-1):
        if inc:
            if price[i+1] < price[i]:
                local.append(i)
                inc = False
        else:
            if price[i+1] > price[i]:
                local.append(i)
                inc = True
    local.append(len(price)-1)
    return local