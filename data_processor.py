import numpy as np

def parse(file):
    f = open(file, 'r')
    price_data = []
    for line in f:
        tok = line.split(',')[1:] # first item is timestamp
        price_data.append([float(i) for i in tok])
    return price_data

def parse_file(data_location):
    file = open(data_location, 'r')
    price_data = []
    for line in file:
        price_data.append(float(line.split(',')[1].rstrip()))
    return price_data

def write_label(label_file, price_file, labels, input_length):
    f = open(label_file, 'w')
    data = [price_file+'\n', str(input_length)+'\n', ','.join(list(map(str,labels)))]
    f.writelines(data)

def read_label(label_file):
    f = open(label_file, 'r')
    data = f.readlines()
    price = parse_file(data[0].strip('\n'))
    return price, int(data[1].strip('\n')), list(map(int, data[2].split(',')))

def get_batch_data(arg, price):
    x = []
    for i in range(arg.input_length - 1, len(price) - 1):
        x.append(price[i-arg.input_length+1:i+1])
    y = get_label_pressure(price[arg.input_length-1:], arg.price_epsilon)
    x = np.array(x)
    y = np.array(y)
    x = x - np.expand_dims(np.average(x, axis=1), axis=1)#/np.std(x)  #centering
    overflow = x.shape[0] % arg.batch_size
    x = np.delete(x, range(x.shape[0]-1-overflow, x.shape[0]-1), axis=0)
    y = np.delete(y, range(y.shape[0]-1-overflow, y.shape[0]-1), axis=0)
    num_bin = len(x) / arg.batch_size
    return np.split(x, num_bin), np.split(y, num_bin)

# percent move
def get_label_complex(price):
    label = []
    for i in range(len(price) - 1):
        ochl_a = np.average(price[i])
        ochl_b = np.average(price[i+1])
        label.append((ochl_b - ochl_a) / ochl_a)
    return label

def get_label_pressure(price, price_epsilon, hold=0, sell=-1, buy=1):
    label = get_label(price, price_epsilon, hold, sell, buy)
    for i in range(len(label)):
        if label[i] == hold:
            for j in reversed(range(i)):
                if label[j] != hold:
                    label[i] = label[j]
                    break
    return label

def get_label(price, price_epsilon, hold, sell, buy):
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
    for i in range(len(price)-1):
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
    for i in range(1, len(price)-1):
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