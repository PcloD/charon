#############################################################################
#PARSING PRICE FILES
#raw columns are:
# Date | Time | Open | High | Low | Close | Volume (BTC) | Volume(Currency) | Weighted Price

from numpy import loadtxt,zeros

def parse(data_location):
    raw=loadtxt(data_location,dtype=str)
    dlength=raw.shape[0]            #total length of table
    data=zeros([dlength,raw.shape[1]-2])
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
    raw=loadtxt(data_location,dtype=str)
    dlength=raw.shape[0] 
    data=zeros([dlength,raw.shape[1]-2])
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

def write_label(label_file, price_file labels):
    f = open(file, 'w')
    data = [price_file, ','.join(labels)]
    f.writelines(data)
