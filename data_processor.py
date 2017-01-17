import numpy as np

def parse_high_frequency(file):
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
    x = get_features_high_frequency(price[:-1])
    y = get_label_high_frequency(price)

    assert len(x) == len(y)

    # change to numpy array
    x = np.array(x)
    y = np.array(y)

    # binning
    overflow = x.shape[0] % arg.batch_size
    x = np.delete(x, range(x.shape[0]-1-overflow, x.shape[0]-1), axis=0)
    y = np.delete(y, range(y.shape[0]-1-overflow, y.shape[0]-1), axis=0)
    num_bin = len(x) / arg.batch_size
    x_split, y_split = np.split(x, num_bin), np.split(y, num_bin)
    pivot = int(len(x_split) / (1-arg.test_size))
    return x_split[:pivot], x_split[pivot:], y_split[:pivot], y_split[pivot:]

def get_features_high_frequency(data):
    """ data t-by-4 matrix or list """
    output_feature = []
    for datum in data:
        o,c,h,l,vol = datum
        feature = []
        output_feature.append(feature)
        avg = np.mean(datum)
        feature.append(avg)
        feature.append(vol)
        feature.append((c-o) / avg)
        feature.append((h-avg) / avg)
        feature.append((avg-l) / avg)
    return output_feature

# percent move
def get_label_high_frequency(price):
    label = []
    for i in range(len(price) - 1):
        ochl_a = np.mean(price[i])
        ochl_b = np.mean(price[i+1])
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