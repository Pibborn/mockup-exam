import pandas
import numpy as np

def load_data(path):
    dataframe = pandas.read_csv(path, header=None)
    temp = []
    for d in dataframe.values:
        if '?' in d:
            np.place(d, d=='?', np.nan)
        temp.append(d)
    return np.array(temp)

def shuffle_data(a, b):
    assert np.shape(a)[0] == np.shape(b)[0]
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    data = load_data('breast-cancer-wisconsin.data')
    print(data)
