import pandas as pd
from scipy.signal import find_peaks
import numpy as np


def find_peaks_larger_than_3(data):
    # 找到波峰的索引位置和波峰的属性
    peaks, properties = find_peaks(-data, height=3)
    return peaks, properties['peak_heights']

def determine_category(peaks):
    if peaks.size == 0:
        category = 0
    else:
        if peaks.size == 1:
            category = 1
        else:
            category = 2
    return category
def main():
    data = np.load('data/y1.npy')
    category = np.zeros((data.shape[0],1))
    for i in range(0, data.shape[0]):
        peaks, _ = find_peaks(-data[i, :], height=3)
        category[i] = determine_category(peaks)

    print(category)
    np.save('data/category.npy', category)

# def main():
#     pass
def trans_csv_to_npy(csv_path, npy_path):
    data = pd.read_csv(csv_path, header=None)
    data = data.values
    np.save(npy_path, data)

if __name__ == '__main__':
    # main()
    trans_csv_to_npy('data/onehot_4classes.csv', 'data/onehot_4classes.npy')
    # data=np.load('data/onehot_15classes.npy')
    # print(data.size)